import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os

from sklearn.preprocessing import MinMaxScaler




import clip
import torch
from torchmetrics.multimodal import CLIPScore
from PIL import Image


df = pd.read_csv("./prompts/memorized_laion_prompts.csv", sep=";")
prompts = df["Caption"].tolist()


# 1. Load the CLIP Model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

f_scores = []
b_scores = []

image_folder = "./generated_images_orig_blocked_v1_4_1_foreground"
for i, prompt in enumerate(prompts):
  break
  for j in range(5):
    

    img = f"{image_folder}/img_{str(i).zfill(4)}_{str(j).zfill(2)}.jpg"
    try:
        image = Image.open(img)
    except FileNotFoundError:
        continue
    # 2. Encode the Prompt
    text_inputs = clip.tokenize([prompt]).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_inputs)

    # 3. Encode the Image
    image_input = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image_input)

    # 4. Normalize Embeddings
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    # 5. Compute Cosine Similarity (Normalized CLIP Score)
    similarity = (image_features @ text_features.T).item()
    f_scores.append(similarity)
    print(f"Normalized CLIP Score: {similarity}")

image_folder = "./generated_images_orig_blocked_v1_4_1_background"

for i, prompt in enumerate(prompts):
  break
  for j in range(0, 0):
    
    img = f"{image_folder}/img_{str(i).zfill(4)}_{str(j).zfill(2)}.jpg"

    try:
        image = Image.open(img)
    except FileNotFoundError:
        continue
    # 2. Encode the Prompt
    text_inputs = clip.tokenize([prompt]).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_inputs)

    # 3. Encode the Image
    image_input = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image_input)

    # 4. Normalize Embeddings
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    # 5. Compute Cosine Similarity (Normalized CLIP Score)
    similarity = (image_features @ text_features.T).item()
    b_scores.append(similarity)
    print(f"Normalized CLIP Score: {similarity}")

scaler = MinMaxScaler()
#scaler.fit(np.array(f_scores + b_scores).reshape(-1, 1))
#f_scores = scaler.transform(np.array(f_scores).reshape(-1, 1)).reshape(-1).tolist()
#b_scores = scaler.transform(np.array(b_scores).reshape(-1, 1)).reshape(-1).tolist()


print(f"Mean Normalized CLIP Score: {np.mean(f_scores)}")
print(f"Mean Normalized CLIP Score: {np.mean(b_scores)}")

# Training foreground: 0.300
# Training Background: 0.287

# memorized foreground: 0.283
# memorized background: 0.268

# mitigated foreground: 0.300
# mitigated background: 0.269

fig, ax = plt.subplots(layout='constrained')

title = "Semantic Alignment Between Background and Foreground of Generated Images"
legend = ["Foreground Component", "Background Component"]
x = np.arange(3)
width = 0.4
x_axis = ["Memorized Images", "No Mitigation", "NeMo"]
y_axis = [0.300, 0.283, 0.300]
y2_axis = [0.287, 0.268, 0.269]

ax.bar(x - 0.2, y_axis, width=0.4, label=legend[0])
ax.bar(x + 0.2, y2_axis, width=0.4, label=legend[1])
ax.set_ylim(0, 0.35)
ax.set_ylabel("CLIP Score")
ax.set_xticks(x, x_axis)
ax.set_title(title)
ax.grid(True)
ax.legend()
fig.savefig("clip_scores_memorized_vs_mitigated.png")

