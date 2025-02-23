import subprocess
import httpx
import time
import json
import os
import logging

logger = logging.getLogger("Ollama")

class Ollama:
        def __init__(self):
                os.environ["OLLAMA_HOST"] = "127.0.0.1:11435"

        def call_ollama(self, text : str) -> dict:
                url = "http://localhost:11435/api/generate"
                data = {
                        "model": "llama3.2",
                        "prompt": text,
                        "stream": False,
                }

                try:
                        print(os.environ["OLLAMA_HOST"])
                        subprocess.run(["echo", "$OLLAMA_HOST"])
                        subprocess.Popen(["ollama", "serve"],
                                stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL)
                        print("Started Ollama")
                        logger.info("Ollama server started.")
                        time.sleep(5)

                        response = httpx.Client().get(url, timeout=5.0)
                        response.raise_for_status()
                        return response.json()
                except Exception:
                        logger.info("An error occurred attempting to request from local ollama")
                        return None
                finally: # Need to have this happen to avoid having random network connection
                        subprocess.run(["pkill", "ollama"])
