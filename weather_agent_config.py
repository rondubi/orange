from generic_agent_inputs import GenericAgentInputs

def get_weather_agent():
        res = GenericAgentInputs()
        res.name = "seven_day_forecast"
        res.extract_args_prompt = """
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
        res.extracted_arg_name = "location"
        res.tools_prompt = """
                You are a helpful weather assistant.
                Given a location and a user's request, use your tools to fulfill the request.
                Only use tools if needed. If you use a tool, make sure the longitude is correctly negative or positive
                Provide a short, concise answer that uses emojis.
                """
        res.tool_api_base = "https://api.open-meteo.com/v1/forecast?current=temperature_2m,precipitation,weather_code&daily=weather_code,temperature_2m_max,temperature_2m_min,precipitation_probability_max&temperature_unit=fahrenheit&wind_speed_unit=mph&precipitation_unit=inch&timezone=America%2FLos_Angeles"
        res.tool_description = {
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
        res.request_fstring = "{{api_base}}&latitude={{latitude}}&longitude={{longitude}}"
        res.request_headers = {"User-Agent": "weather-app/1.0", "Accept": "application/geo+json"}

        def p(data : dict):
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

        res.process_results = p

        return res
