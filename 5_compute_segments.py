import PIL
from PIL import Image
import pandas as pd

from rembg import new_session, remove

import argparse
from skimage.metrics import structural_similarity
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
import torch

from torchvision import transforms

import os

normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],
)
small_288 = transforms.Compose([
    transforms.Resize(288),
    transforms.ToTensor(),
    normalize,
])
skew_320 = transforms.Compose([
    transforms.Resize([320, 320]),
    transforms.ToTensor(),
    normalize,
])

def fb_strip(session, im):
    foreground = remove(im,
        session=session,
        alpha_matting=True,
        alpha_matting_foreground_threshold=270,
        alpha_matting_background_threshold=20,
        alpha_matting_erode_size=11)

    alpha_chan = foreground.split()[-1].convert("L")
    #inverted_alpha = PIL.ImageOps.invert(alpha)

    background = im.copy()
    background.paste(alpha_chan, mask=alpha_chan)

    return foreground, background

def calculate_score(image1, image2):
    image1 = image1.resize((512,512))
    image2 = image2.resize((512,512))
    img1 = np.asarray(image1)#.convert("L"))
    img2 = np.asarray(image2)#.convert("L"))

    #print(img1.shape)  # prints (30, 60, 3)
    #print(img2.shape)  # prints (30, 60, 3)

    measure_value = structural_similarity(img1, img2, win_size=3) #, multichannel=True)
    return measure_value

def calculate_sscd(image1, image2):
  model_sscd = torch.jit.load("./sscd_disc_mixup.torchscript.pt")
  img = image1.convert('RGB')
  img2 = image2.convert('RGB')
  batch = small_288(img).unsqueeze(0)
  batch2 = small_288(img2).unsqueeze(0)
  embedding = model_sscd(batch)[0, :]
  embedding2 = model_sscd(batch2)[0, :]
  return embedding.dot(embedding2).item()


def compute(g_fold, simple_name=False, sscd=False):

    model_name = "birefnet-general"
    session = new_session(model_name, providers=['CUDAExecutionProvider'])

    num_images = 500

    mem_folder = "memorized_images/"

    for i in range(365, 366):
        print(i)
        im_name = "{}.jpg".format(str(i))
        try:

          mem_image = Image.open(mem_folder + im_name)
        except:
          continue

        #print(mem_image)

        if not sscd:
          mem_foreground, mem_background = fb_strip(session, mem_image)

        mem_foreground.save("memorized_images/foreground/{}.png".format(str(i)))
        mem_background.save("memorized_images/background/{}.png".format(str(i)))
    return

def main(args):
    os.environ["OMP_NUM_THREADS"] = "2"
    compute(args.sname)

    print("Finished")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Add an argument
    parser.add_argument('--sname', type=str, required=False)
    # Parse the argument
    args = parser.parse_args()    
    main(args)