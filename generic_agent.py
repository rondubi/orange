# This is a basic agent that uses Mistral AI to answer weather questions.
# This agent is designed to be piped every single message in a Discord server.
# First, the agent checks for a location in the message, and extracts it if it exists.
# This prevents the agent from responding to messages that don't ask about weather.
# Then, a separate prompt chain is used to get the weather data and response.

import os
import json
import logging
import discord
import httpx
from dataclasses import dataclass
from jinja2 import Environment

from mistralai import Mistral

logger = logging.getLogger("discord")
MISTRAL_MODEL = "mistral-large-latest"

jenv = Environment()

@dataclass
class GenericAgentInputs:
        name                : str = None
        extract_args_prompt : str = None
        extracted_arg_name  : str = None
        tools_prompt        : str = None
        tool_api_base       : str = None
        tool_description    : dict= None # json
        request_fstring     : str = None
        request_headers     : dict= None # json

inputs = GenericAgentInputs()
inputs.name = "seven_day_forecast"
inputs.extract_args_prompt = """
        Is this message explicitly requesting weather information for a specific city/location?
        If not, return {"location": "none"}.

        Otherwise, return the full name of the city in JSON format.

        Example:
        Message: What's the weather in sf?
        Response: {"location": "San Francisco, CA"}

        Message: What's the temperature in nyc?
        Response: {"location": "New York City, NY"}

        Message: Is it raining in sf?
        Response: {"location": "San Francisco, CA"}

        Message: I love the weather in SF
        Response: {"location": "none"}
        """
inputs.extracted_arg_name = "location"
inputs.tools_prompt = """
        You are a helpful weather assistant.
        Given a location and a user's request, use your tools to fulfill the request.
        Only use tools if needed. If you use a tool, make sure the longitude is correctly negative or positive
        Provide a short, concise answer that uses emojis.
        """
inputs.tool_api_base = "https://api.open-meteo.com/v1/forecast?current=temperature_2m,precipitation,weather_code&daily=weather_code,temperature_2m_max,temperature_2m_min,precipitation_probability_max&temperature_unit=fahrenheit&wind_speed_unit=mph&precipitation_unit=inch&timezone=America%2FLos_Angeles"
inputs.tool_description = {
        "type": "function",
        "function": {
                "name": "seven_day_forecast",
                "description": "Get the seven day forecast for a given location with latitude and longitude.",
                "parameters": {
                        "type": "object",
                        "properties": {
                                "latitude": {"type": "string"},
                                "longitude": {"type": "string"},
                        },
                        "required": ["latitude", "longitude"],
                },
        },
        }
inputs.request_fstring = "{api_base}&latitude={latitude}&longitude={longitude}"
inputs.request_headers = {"User-Agent": "weather-app/1.0", "Accept": "application/geo+json"}

class GenericAgent:
        def __init__(self):
                MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

                self.client = Mistral(api_key=MISTRAL_API_KEY)
                self.tools = [ inputs.tool_description ]
                self.tools_to_functions = {
                        inputs.name: use_tool_and_clean_results,
                }

        async def extract_arg(self, message: str) -> dict:
                logger.info("Extracting arg")
                # Extract the location from the message.
                response = await self.client.chat.complete_async(
                        model=MISTRAL_MODEL,
                        messages=[
                                {"role": "system", "content": inputs.extract_args_prompt},
                                {"role": "user", "content": f"Discord message: {message}\nOutput:"},
                        ],
                        response_format={"type": "json_object"},
                )

                logger.info("Made request to extract argument from user message")

                message = response.choices[0].message.content

                obj = json.loads(message)
                if obj[inputs.extracted_arg_name] == "none":
                        return None

                return obj[inputs.extracted_arg_name]

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
                tool_response = await self.client.chat.complete_async(
                        model=MISTRAL_MODEL,
                        messages=messages,
                        tools=self.tools,
                        tool_choice="any",
                )

                messages.append(tool_response.choices[0].message)

                tool_call = tool_response.choices[0].message.tool_calls[0]
                function_name = tool_call.function.name
                
                logger.info(f"Function name is {function_name}")

                function_params = json.loads(tool_call.function.arguments)
                function_result = self.tools_to_functions[function_name](**function_params)

                # Append the tool call and its result to the messages.
                messages.append(
                        {
                                "role": "tool",
                                "name": function_name,
                                "content": function_result,
                                "tool_call_id": tool_call.id,
                        }
                )

                logger.info(f"Messages: {messages}")

                # Run the model again with the tool call and its result.
                response = await self.client.chat.complete_async(
                        model=MISTRAL_MODEL,
                        messages=messages,
                )

                return response.choices[0].message.content

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
        try:
                response = httpx.Client().get(url, headers=inputs.headers, timeout=5.0)
                response.raise_for_status()
                return response.json()
        except Exception:
                return None


def use_tool_and_clean_results(latitude : str, longitude : str):
        logger.info("Using tool and cleaning results")
        url = jenv.from_string(inputs.request_fstring).render({
                "api_base" : inputs.tool_api_base,
                "latitude" : latitude,
                "longitude" : longitude
        })

        data = _make_request(url)

        if data is None:
                return "Error fetching data"


        res_json = {
                "current": data["current"],
                "daily": {},
        }

        for i, time in enumerate(data["daily"]["time"]):
                max_temp = data["daily"]["temperature_2m_max"][i]
                min_temp = data["daily"]["temperature_2m_min"][i]
                precipitation = data["daily"]["precipitation_probability_max"][i]
                res_json["daily"][time] = {
                        "weather_code": data["daily"]["weather_code"][i],
                        "temperature_max": f"{max_temp}°F",
                        "temperature_min": f"{min_temp}°F",
                        "precipitation": f"{precipitation}%",
                }

        return json.dumps(res_json)

