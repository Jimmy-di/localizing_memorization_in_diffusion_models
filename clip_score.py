import torch
import clip
from PIL import Image
import os

def get_clip_score(image_path, text):
# Load the pre-trained CLIP model and the image
    model, preprocess = clip.load('ViT-B/32')
    image = Image.open(image_path)

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
    
    # Calculate the cosine similarity to get the CLIP score
    clip_score = torch.matmul(image_features, text_features.T).item()
    
    return clip_score

scores = []
foldername = "generated_images_orig_blocked_r_cluster_0_block_all"

text = [
    "Wilderness Abstract - Throw Pillow",
    "Sapphire & Jade Stained … Throw Pillow",
    "Geothermal Area With Steaming Hot Springs In Iceland - Throw Pillow",
    "Intellectual Dark Web - Unix Reboot (black background) Home Throw Pillow by Mythic Ink's Shop",
    "Saxo Turquoise  - Throw Pillow"
        ]

for i in range(18, 23):
    for j in range(5):
        image_name = "img_00" + str(i) + "_0" + str(j) + ".jpg"
        filename = os.path.join(foldername, image_name)
        score = get_clip_score(filename, text[i-18])
        scores.append(score)
            


print(f"CLIP Score: {sum(scores) / len(scores)}")