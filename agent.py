import os
import logging
import random
from PIL import Image
import io
import discord
from mistralai import Mistral
from caption import caption_template
from meme_grabber import MemeGrabber
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("discord")

MISTRAL_MODEL = "mistral-large-latest"

EXTRACT_MEME_SOURCE_PROMPT = """
You are a meme creation expert.
When a user indicates in a message that they want to create a meme,
you determine the best source of the three sources to find the meme that we should use for
the message.

The three sources are:
- imgflip: This source retrieves meme templates from ImgFlip, a well-known meme generator and database. It includes a collection of around 200 classic and widely recognized meme formats, making it ideal for timeless and easily recognizable memes.
- reddit: This source pulls meme templates from Reddit, leveraging a semantic search for contemporary and trending meme formats. Since these templates come directly from user-generated content, they may already contain text and reflect the latest internet humor and cultural references.
- reaction: This source gets images from the static source of meme reactions we have, which contains three large memes categories: 1) angry / wicked memes 2) no / stop / police memes 3) WTF type of memes.

You extract the canonical name of the meme template,
as well as the top and bottom captions to place on the meme,
and return them in a Python-style dict of strings.
Keep the top and bottom captions concise,
and make sure that if combined they still read as a normal sentence.

Otherwise, return an empty dict like {}.

If there are not indications on trying to create a meme, return an empty dict like {}.

Example:
Message: Make a condescending Wonka meme captioned 'tell me more about your GPT wrapper startup'
Response: {"source" : "imgflip", "template": "condescending wonka", "top" : "Tell me more", "bottom" : "about your GPT wrapper startup"}

Message: Make an frustrated Tom meme about when your group doesn't have the 153 project done the day before the deadline.
Response: {"source" : "reaction", "template": "Angry - Wicked", "top" : "When your group has a day to do", "bottom" : "the 153 project"}

Message: Make a concerned Tom meme about when your group doesn't have the 153 project done the day before the deadline.
Response: {"source" : "imgflip", "template": "concerned", "top" : "When your group has a day to do", "bottom" : "the 153 project"}

Message: Make a meme of exit 12 off ramp saying "when they tell me the project isn't it C"
Response: {"source" : "imgflip", "template": "exit 12", "top": "when they tell me the project isn't", "bottom": "in C"}

Message: Make a meme of marvel character saying "when they tell me the project isn't it C"
Response: {"source" : "reddit", "template": "marvel character", "top": "when they tell me the project isn't", "bottom": "in C"}

Message: I like memes
Response: {}

Message: just trying new things
Response: {}
"""

grabber = MemeGrabber(
    reddit_client_id=os.getenv("YOUR_REDDIT_CLIENT_ID"),
    reddit_client_secret=os.getenv("YOUR_REDDIT_CLIENT_SECRET"),
    reddit_user_agent=os.getenv("REDDIT_USER_AGENT"),
    local_memes_folder=os.getenv("REAL_REACTIONS_DIR", "/default/path")
)

class MistralAgent:
    def __init__(self):
        MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

        self.client = Mistral(api_key=MISTRAL_API_KEY)

    async def run(self, message: discord.Message):
        messages = [
            {"role": "system", "content": EXTRACT_MEME_SOURCE_PROMPT},
            {"role": "user", "content": message.content},
        ]
        
        logger.info(f"Messages are {messages}")

        response = await self.client.chat.complete_async(
            model=MISTRAL_MODEL,
            messages=messages,
        )
        
        logger.info(f"Response is {response}")
        extracted_args = response.choices[0].message.content
        logger.info(f"Extracted arguments {extracted_args}")

        # {"source": [either imgflip, reddit, reaction], "template" : [some url here], 
        #             "top" : [top caption], "bottom", [bottom caption]}

        extracted_dict = \
                eval(extracted_args[extracted_args.find('{') : extracted_args.rfind('}') + 1])
            
        logger.info(f"Extracted dict {extracted_dict}")
        
        # Not a query for a meme
        if (not extracted_dict):
            return "INSUFFICIENT_MESSAGE"

        # Call function to get template url
        urls = grabber.get_template(
            source = extracted_dict["source"],
            query = extracted_dict["template"],
            limit = 1
        )

        logger.info(f"Urls are {urls}")

        if (not urls["success"]):
            return "INSUFFICIENT_MESSAGE"

        # Call function to caption
        if (extracted_dict["source"] == "reaction"):
            image_bytes = await caption_template(urls["local_path"], extracted_dict["bottom"], \
                extracted_dict["top"], "png")
        else:
             image_bytes = await caption_template(urls["templates"][0]["url"], extracted_dict["bottom"], \
                extracted_dict["top"], "png")
        
        output_path = f"{random.getrandbits(24)}.png"
        img = Image.open(io.BytesIO(image_bytes))
        img.save(output_path, "PNG")

        # return image filename
        return output_path