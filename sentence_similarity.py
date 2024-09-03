import json
import requests

API_URL = "https://api-inference.huggingface.co/models/sentence-transformers/msmarco-distilbert-base-tas-b"
api_token = "hf_"
headers = {"Authorization": "Bearer {}".format(api_token)}

def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

data = query(
    {
        "inputs": {
            "source_sentence": "If Barbie Were The Face of The World's Most Famous Paintings",
            "sentences": [
                "Barbie as The Face of The World's Most Famous Paintings",
                "The World's Most Famous Paintings with the face of Barbie",
                "The World's Most Famous Paintings with Barbie's face",
                "World Famous Paintings but with the face of Barbie"
            ]
        }
    }
)
print(data)
## [0.9907169342041016, 0.9841938614845276, 0.9787886142730713, 0.9771206974983215] 
