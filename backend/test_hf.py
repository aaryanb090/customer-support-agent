import os
import requests


token = os.getenv("HF_API_TOKEN")
if not token:
    raise RuntimeError("HF_API_TOKEN not set – make sure you did `set HF_API_TOKEN=…` in this shell")


url = "https://api-inference.huggingface.co/models/facebook/bart-large-mnli"
headers = {"Authorization": f"Bearer {token}"}
payload = {
    "inputs": "How do I reset my password?",
    "parameters": {
        "candidate_labels": [
            "technical support request",
            "product feature suggestion",
            "sales inquiry"
        ]
    }
}

resp = requests.post(url, headers=headers, json=payload)
resp.raise_for_status()  


print(resp.json())
