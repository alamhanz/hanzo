# Querying chat models with Together AI
import os

from dotenv import load_dotenv
from langchain_together import ChatTogether

# from dotenv import load_dotenv

load_dotenv()
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")

# choose from our 50+ models here: https://docs.together.ai/docs/inference-models
chat = ChatTogether(
    together_api_key=TOGETHER_API_KEY,
    model="meta-llama/Llama-3-70b-chat-hf",
)

# stream the response back from the model
for m in chat.stream("Tell me fun things to do in Jakarta, Indonesia"):
    print(m.content, end="", flush=True)

# if you don't want to do streaming, you can use the invoke method
# chat.invoke("x")
