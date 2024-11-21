import torch
from utils.stable_diffusion import load_sd_components, load_text_components, generate_images
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
"""
source_sentence = ["Emma Watson to play Belle in Disney's <i>Beauty and the Beast</i>"]            # Prompt 0
sentences = ["Emma Watson Cast as Belle in Disney’s Beauty and the Beast",
"Emma Watson Will Portray Belle in Disney’s Beauty and the Beast",
"Disney Selects Emma Watson to Play Belle in Beauty and the Beast",
"Emma Watson Takes on the Role of Belle in Disney’s Beauty and the Beast",
"Emma Watson to Star as Belle in Disney’s Live-Action Beauty and the Beast",
"Belle to Be Played by Emma Watson in Disney’s Beauty and the Beast",
"Emma Watson Chosen for Belle Role in Disney’s Beauty and the Beast",
"Emma Watson Will Bring Belle to Life in Disney’s Beauty and the Beast",
"Disney’s Beauty and the Beast to Feature Emma Watson as Belle"]
"""


source_sentence = ["George R.R. Martin to Focus on Writing Next Book, World Rejoices"]            # Prompt 0
sentences = ["George R.R. Martin Shifts Focus to Writing Next Book, Fans Celebrate",
"George R.R. Martin Prioritizes Next Novel, Bringing Joy to Fans",
"World Rejoices as George R.R. Martin Commits to Finishing His Next Book",
"George R.R. Martin Dedicates Time to Writing Next Book, Delighting Readers",
"Fans Rejoice as George R.R. Martin Focuses on Completing Upcoming Novel",
"George R.R. Martin Turns Full Attention to Next Book, to the Delight of Fans",
"Next Book Takes Priority for George R.R. Martin, Sparking Worldwide Excitement",
"George R.R. Martin Plans to Finish His Next Novel, and the World Celebrates",
"Worldwide Cheers as George R.R. Martin Focuses Solely on Writing His Next Book"]


version = 'v1-4'

tokenizer, text_encoder = load_text_components(version)
torch_device = "cuda"
text_encoder.to(torch_device)

source_input = tokenizer(source_sentence,
                            padding="max_length",
                            max_length=tokenizer.model_max_length,
                            truncation=True,
                            return_tensors="pt")
source_embedding = text_encoder(
    source_input.input_ids.to(text_encoder.device))[0].reshape(1, -1)

#Sanity check
output = cosine_similarity(source_embedding.cpu().detach().numpy(), source_embedding.cpu().detach().numpy())[0]
print(output)
for prompt in sentences:
    text_input = tokenizer([prompt],
                            padding="max_length",
                            max_length=tokenizer.model_max_length,
                            truncation=True,
                            return_tensors="pt")
    text_embedding = text_encoder(
        text_input.input_ids.to(text_encoder.device))[0].reshape(1, -1)

    #cosine_sim = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
    
    output = cosine_similarity(source_embedding.cpu().detach().numpy(), text_embedding.cpu().detach().numpy())[0]
    print(output)
