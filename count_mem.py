import pandas as pd
import numpy as np
import PIL
from PIL import Image
import os, os.path
from os import mkdir, path, getcwd

def get_subdirectory(sd):
    dir = path.join(getcwd(), sd)
    if not path.isdir(dir):
        mkdir(dir)
    return dir

clustered_prpt = "prompts/memorized_laion_prompts_cluster_all_embeddings.csv"
image_folder = "./memorized_images/"

prompts = pd.read_csv(clustered_prpt, index_col=False, sep=';')

for i in range(500):
    try:
        img = Image.open(image_folder+"{}.jpg".format(str(i)))
    except:
        continue

    print(str(prompts.iloc[i]['cluster']))
    dname=get_subdirectory(image_folder + 'clustered/'+str(prompts.iloc[i]['cluster']))

    cp = pd.read_csv("prompts/memorized_laion_prompts_cluster_{}_embeddings.csv".format(str(prompts.iloc[i]['cluster'])), index_col=False, sep=';')

    target_url = prompts.iloc[i]['URL']
    result_index = cp.loc[cp['URL'] == target_url].index +2
    print(result_index)
    #image save example
    img.save(f"{dname}/{result_index[0]}.jpg")
