import time
class OpenAIWrapper:
    def __init__(self, api_key=None, retry_wait_time=5):
        self.client = openai.OpenAI(api_key=api_key)
        self.retry_wait_time = retry_wait_time

    def chat_completion_with_retry(self, messages, model, retries=2):
        while retries > 0:
            try:
                response = self.client.chat.completions.create(
                    messages=messages,
                    model=model,
                    temperature=0
                )
                return response.choices[0].message.content.lower().strip()
            except Exception as e:
                print(f"An error occurred: {e}. Retrying...")
                retries -= 1
                time.sleep(self.retry_wait_time)
        return "skipped"