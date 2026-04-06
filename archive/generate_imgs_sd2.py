from pathlib import Path
from typing import Tuple, Union, Optional
from urllib.request import urlretrieve

from matplotlib import colors
import matplotlib.pyplot as plt
import numpy as np
import requests
import torch
import tqdm
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

import pandas as pd
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

num_imgs = 5
seed = 2
torch.manual_seed(2)

model_id = "stabilityai/stable-diffusion-2-1"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)

pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to("cuda")

prompts = pd.read_csv("prompts/memorized_laion_prompts.csv", sep=';')
dname = "generated_images_orig_unblocked_v2_1_new/"
for index, row in prompts.iterrows():

    prompt=row['Caption']
    
    for i in range(num_imgs):
        image = pipe(prompt).images[0]
        img_name = "img_{}_{}.jpg".format(str(index), str(i))
        image.save(dname+img_name)