import os
import logging
import random
from PIL import Image
import io

from mistralai import Mistral
import discord
from caption import caption_template
from meme_grabber import MemeGrabber

logger = logging.getLogger("discord")

MISTRAL_MODEL = "mistral-large-latest"

EXTRACT_MEME_ARGUMENTS_PROMPT = """
You are a meme creation expert.
When a user indicates in a message that they want to create a meme,
you extract the canonical name of the meme template,
as well as the top and bottom captions to place on the meme,
and return them in a Python-style dict of strings.
Keep the top and bottom captions concise,
and make sure that if combined they still read as a normal sentence.

Otherwise, return an empty dict.

Example:
Message: Make a condescending Wonka meme captioned 'tell me more about your GPT wrapper startup'
Response: {"template" : "Condescending Wonka", "top" : "Tell me more", "bottom" : "about your GPT wrapper startup"}

Message: Make a concerned Tom meme about when your group doesn't have the 153 project done the day before the deadline.
Response: {"template" : "Concerned Tom", "top" : "When your group has a day to do", "bottom" : "the 153 project"}

Message: I like memes
Response: {}
"""

grabber = MemeGrabber(
    reddit_client_id="YOUR_REDDIT_CLIENT_ID",
    reddit_client_secret="YOUR_REDDIT_CLIENT_SECRET",
    reddit_user_agent="python:meme-grabber:v1.0",
    local_memes_folder="reaction_memes"
)

class MistralAgent:
    def __init__(self):
        MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

        self.client = Mistral(api_key=MISTRAL_API_KEY)

    async def run(self, message: discord.Message):
        # The simplest form of an agent
        # Send the message's content to Mistral's API and return Mistral's response

        messages = [
            {"role": "system", "content": EXTRACT_MEME_ARGUMENTS_PROMPT},
            {"role": "user", "content": message.content},
        ]
        
        logger.info(f"Messages are {messages}")

        response = await self.client.chat.complete_async(
            model=MISTRAL_MODEL,
            messages=messages,
        )

        # TODO @ Ramya: instead of having the next several lines manually do template search,
        # provide the LLM access to the functions Matthew has provided in meme_grabber.py
        #
        # It should then return
        # {"template" : [some url here], "top" : [top caption], "bottom", [bottom caption]}
        #
        # Once that is done, you should be able to get rid of everything before the image
        # construction (aside from a bit of error handling)

        logger.info(f"Response is {response}")
        extracted_args = response.choices[0].message.content
        logger.info(f"Extracted arguments {extracted_args}")

        extracted_dict = \
                eval(extracted_args[extracted_args.find('{') : extracted_args.rfind('}') + 1])

        logger.info(f"Extracted dict {extracted_dict}")
        # Not a query for a meme
        if (not extracted_dict):
                return None

        # Call function to get template url
        urls = grabber.get_template(
            source = "imgflip",
            query = extracted_dict["template"].lower(),
            limit = 1
        )

        logger.info(f"Urls are {urls}")

        if (not urls["success"]):
                return None

        # Call function to caption
        image_bytes = await caption_template(urls["templates"][0]["url"], extracted_dict["bottom"], \
                extracted_dict["top"], "png")
        
        output_path = f"{random.getrandbits(24)}.png"
        img = Image.open(io.BytesIO(image_bytes))
        img.save(output_path, "PNG")

        # return image filename
        return output_path

