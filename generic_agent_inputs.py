from collections.abc import Callable
from dataclasses import dataclass

@dataclass
class GenericAgentInputs:
        name                : str = None
        extract_args_prompt : str = None
        extracted_arg_name  : str = None
        tools_prompt        : str = None
        tool_api_base       : str = None
        tool_description    : dict= None # json
        request_fstring     : str = None # must be jinja fstring, use {{ rather than {
        request_headers     : dict= None # json
        # accepts a results dict representing json, returns a string to show the user
        process_results     : Callable[[dict], str] = None
