import os
from mistralai import Mistral
import discord
from caption import caption_template

MISTRAL_MODEL = "mistral-large-latest"

EXTRACT_MEME_ARGUMENTS_PROMPT = """
You are a meme creation expert.
When a user indicates in a message that they want to create a meme,
you extract the canonical name of the meme template,
as well as the top and bottom captions to place on the meme,
and return them in a Python-style dict of strings.

Otherwise, return an empty dict.

Example:
Message: Make a condescending Wonka meme captioned 'tell me more about your GPT wrapper startup'
Response: {"template" : "Condescending Wonka", "top" : "Tell me more", "bottom" : "about your GPT wrapper startup"}

Message: Make a concerned Tom meme about when your group doesn't have the 153 project done the day before the deadline.
Response: {"template" : "Concerned Tom", "top" : "When your group has a day to do", "bottom" : "and the 153 project"}

Message: I like memes
Response: {}
"""

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

        response = await self.client.chat.complete_async(
            model=MISTRAL_MODEL,
            messages=messages,
        )

        extracted_args = dict(response.choices[0].message.content)
        logger.info(f"Extracted arguments {extracted_args}")

        # Not a query for a meme
        if (not extracted_args):
                return None

        """
        # Call function to get template url
        url = # something

        # Call function to caption
        caption_template()

        # return image filename
        """

