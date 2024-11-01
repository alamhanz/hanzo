import os

from dotenv import load_dotenv
from together import Together

load_dotenv()
# TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")

client = Together()

response = client.chat.completions.create(
    model="Qwen/Qwen2.5-7B-Instruct-Turbo",
    messages=[{"role": "user", "content": "Tell me about earth"}],
    max_tokens=512,
    temperature=0.7,
    top_p=0.7,
    top_k=50,
    repetition_penalty=1,
    stop=["<|im_end|>"],
    stream=True,
)
for token in response:
    if hasattr(token, "choices"):
        print(token.choices[0].delta.content, end="", flush=True)
