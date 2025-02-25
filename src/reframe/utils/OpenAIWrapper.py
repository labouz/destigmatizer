import time
from openai import OpenAI
from .base import LLMClient

class OpenAIWrapper(LLMClient):
    def __init__(self, api_key=None, retry_wait_time=5, default_model="gpt-4o"):
        self.client = OpenAI(api_key=api_key)
        self.retry_wait_time = retry_wait_time
        self.default_model = default_model

    def generate_completion(self, messages, model=None, retries=2, temperature=0):
        model = model or self.default_model
        while retries > 0:
            try:
                response = self.client.chat.completions.create(
                    messages=messages,
                    model=model,
                    temperature=temperature
                )
                return response.choices[0].message.content.lower().strip()
            except Exception as e:
                print(f"An error occurred: {e}. Retrying...")
                retries -= 1
                time.sleep(self.retry_wait_time)
        return "skipped"