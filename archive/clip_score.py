import torch
import clip
from PIL import Image
import os

import json

import PIL
from PIL import Image

import pandas as pd

def get_clip_score(image, text):
# Load the pre-trained CLIP model and the image
    model, preprocess = clip.load('ViT-B/32')

    # Preprocess the image and tokenize the text
    image_input = preprocess(image).unsqueeze(0)
    text_input = clip.tokenize([text])
    
    # Move the inputs to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    image_input = image_input.to(device)
    text_input = text_input.to(device)
    model = model.to(device)
    
    # Generate embeddings for the image and text
    with torch.no_grad():
        image_features = model.encode_image(image_input)
        text_features = model.encode_text(text_input)
    
    # Normalize the features
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    # Calculate the cosine similarity to get the CLIP score
    clip_score = cos(image_features, text_features).item()
    
    return clip_score

scores = []
foldername = "generated_images_orig_unblocked_v1_4_1"

caption_file = "prompts/memorized_laion_prompts.csv"


df = pd.read_csv(caption_file, sep=";")

for i in range(500):
    text_prompt = df['Caption'].iloc[i]
    print(text_prompt)
    for j in range(5):
        if i < 10:
            img_name = "img_000{}_0{}.jpg".format(str(i), str(j))
        elif i < 100:
            img_name = "img_00{}_0{}.jpg".format(str(i), str(j))
        elif i < 1000:
            img_name = "img_0{}_0{}.jpg".format(str(i), str(j))
        else:
            img_name = "{}_{}.jpg".format(str(i), str(j))

        img = Image.open("generated_images_orig_unblocked_v1_4_1" + '/' +img_name)
        clip_score = get_clip_score(img, text_prompt)
        print(clip_score)

        scores.append(clip_score)


with open("generated_images_orig_unblocked_v1_4_1/output.json", "w") as file:
    json.dump(scores, file)

print(max(scores))
print(min(scores))
print(f"CLIP Score: {sum(scores) / len(scores)}")
