import os

import requests
from dotenv import load_dotenv

load_dotenv()

hf_token = os.getenv("hf_token")
model_id = (
    "GoToCompany/gemma2-9b-cpt-sahabatai-v1-base"  ## Doesn't support the inference API
)

# model_id = "sentence-transformers/all-MiniLM-L6-v2"
# model_id = "HIT-TMG/KaLM-embedding-multilingual-mini-instruct-v1"

# api_url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{model_id}"
# headers = {"Authorization": f"Bearer {hf_token}"}


# def query(texts):
#     response = requests.post(
#         api_url,
#         headers=headers,
#         json={"inputs": texts, "options": {"wait_for_model": True}},
#     )
#     return response.json()

api_url = f"https://api-inference.huggingface.co/models/{model_id}"
headers = {"Authorization": f"Bearer {hf_token}"}


def chat(payload):
    response = requests.post(api_url, headers=headers, json=payload)
    return response.json()


# data = {"inputs": "who are you?", "parameters": {"max_length": 50}}
data = {
    "inputs": {
        "question": "who are you?",
        "context": "I want to know about how you have been trained as a model",
    }
}

response = chat(data)
print(response)

# API_URL = "https://api-inference.huggingface.co/models/deepset/roberta-base-squad2"

# data = {
#     "inputs": {
#         "question": "What is the capital of France?",
#         "context": "France's capital is Paris.",
#     }
# }

# response = requests.post(API_URL, headers=headers, json=data)
# print(response.json())
