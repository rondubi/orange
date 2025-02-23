# This is a basic agent that uses Mistral AI to answer weather questions.
# This agent is designed to be piped every single message in a Discord server.
# First, the agent checks for a location in the message, and extracts it if it exists.
# This prevents the agent from responding to messages that don't ask about weather.
# Then, a separate prompt chain is used to get the weather data and response.

import re
import os
import json
import logging
import discord
import httpx
from jinja2 import Environment
from weather_agent_config import get_weather_agent
import ollama

# from mistralai import Mistral

logger = logging.getLogger("discord")
OLLAMA_MODEL = "llama3.2"

jenv = Environment()

inputs = get_weather_agent()

class GenericAgent:
        def __init__(self):
                # MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

                # self.client = Mistral(api_key=MISTRAL_API_KEY)
                self.tools = [ inputs.tool_description ]
                self.tools_to_functions = {
                        inputs.name: use_tool_and_clean_results,
                }

        async def extract_arg(self, message: str) -> dict:
                logger.info("Extracting arg")
                # Extract the location from the message.
                response = ollama.chat(model="llama3.2",
                        messages=[
                                {"role": "system", "content": inputs.extract_args_prompt},
                                {"role": "user", "content": f"Discord message: {message}\nOutput:"},
                        ],
                        # response_format={"type": "json_object"},
                )

                logger.info("Made request to extract argument from user message")
                logger.info(f"Result of request was {response.message.content}")

                result = re.search(r'{"location": (.*)}', response.message.content)
                if result == None:
                        logger.info("Failed to retrieve information")
                        return None

                return result

        async def use_tool(self, arg_string: str, request: str):
                logger.info("Using tool")
                messages = [
                        {"role": "system", "content": inputs.tools_prompt},
                        {
                                "role": "user",
                                "content": f"{inputs.extracted_arg_name}: {arg_string}\n\
                                Request: {request}\nOutput:",
                        },
                ]

                logger.info(f"Messages: {messages}")

                # Require the agent to use a tool with the "any" tool choice.
                tool_response = ollama.chat(
                        model=OLLAMA_MODEL,
                        messages=messages,
                        tools=self.tools,
                        # tool_choice="any",
                )

                logger.info(f"On using tools, result was {tool_response}")

                messages.append(tool_response.message)

                tool_call = tool_response.message.tool_calls[0]
                function_name = tool_call.function.name
                
                logger.info(f"Function name is {function_name}")

                # function_params = json.loads(tool_call.function.arguments)
                function_result = self.tools_to_functions[function_name](tool_call.function.arguments)

                # Append the tool call and its result to the messages.
                messages.append(
                        {
                                "role": "tool",
                                "name": function_name,
                                "content": function_result,
                                # "tool_call_id": tool_call.id,
                        }
                )

                logger.info(f"Messages: {messages}")

                # Run the model again with the tool call and its result.
                response = ollama.chat(
                        model=OLLAMA_MODEL,
                        messages=messages,
                )

                return response.message.content

        async def run(self, message: discord.Message):
                logger.info("Running")
                # Extract the argument from the message to verify that the user is asking us to do that
                arg = await self.extract_arg(message.content)
                if arg is None:
                        return None

                # Send a message to the user that we are using the tool
                pending_message = await message.reply("Working...")

                # Use a second prompt chain to get the weather data and response.
                response = await self.use_tool(arg, message.content)

                # Edit the message to show the weather data.
                await pending_message.edit(content=response)


def _make_request(url: str):
        logger.info(f"Making request to {url}")
        try:
                response = httpx.Client().get(url, headers=inputs.request_headers, timeout=50.0)
                response.raise_for_status()
                return response.json()
        except Exception as e:
                logger.info(f"Request failed because {e}")
                return None


def use_tool_and_clean_results(arguments : dict):
        logger.info("Using tool and cleaning results")
        url = jenv.from_string(inputs.request_fstring).render({
                "api_base" : inputs.tool_api_base,
                **arguments
        })

        data = _make_request(url)
        logger.info(f"Data: {data}")

        if data is None:
                return "Error fetching data"

        res = inputs.process_results(data)
        logger.info(f"Returning {res}")
        return res
