import PIL
from PIL import Image
import pandas as pd

from rembg import new_session, remove

import argparse
from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure
from torchmetrics import StructuralSimilarityIndexMeasure
import numpy as np

from queue import PriorityQueue
import seaborn as sns
import matplotlib.pyplot as plt
import torch

from torchvision import transforms

import os

normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
)
small_288 = transforms.Compose(
    [
        transforms.Resize(288),
        transforms.ToTensor(),
        normalize,
    ]
)
skew_320 = transforms.Compose(
    [
        transforms.Resize([320, 320]),
        transforms.ToTensor(),
        normalize,
    ]
)


def fb_strip(session, im):
    foreground = remove(
        im,
        session=session,
        alpha_matting=True,
        alpha_matting_foreground_threshold=270,
        alpha_matting_background_threshold=20,
        alpha_matting_erode_size=11,
    )

    alpha_chan = foreground.split()[-1].convert("L")
    # inverted_alpha = PIL.ImageOps.invert(alpha)

    background = im.copy()
    background.paste(alpha_chan, mask=alpha_chan)

    return foreground, background


def calculate_score(image1, image2):
    image1 = image1.resize((512, 512))
    image2 = image2.resize((512, 512))
    img1 = np.asarray(image1)  # .convert("L"))
    img2 = np.asarray(image2)  # .convert("L"))

    # print(img1.shape)  # prints (30, 60, 3)
    # print(img2.shape)  # prints (30, 60, 3)

    measure_value = MultiScaleStructuralSimilarityIndexMeasure(
        img1, img2, 
    )  # , multichannel=True)
    return measure_value


def calculate_sscd(image1, image2):
    model_sscd = torch.jit.load("./sscd_disc_mixup.torchscript.pt")
    img = image1.convert("RGB")
    img2 = image2.convert("RGB")
    batch = small_288(img).unsqueeze(0)
    batch2 = small_288(img2).unsqueeze(0)
    embedding = model_sscd(batch)[0, :]
    embedding2 = model_sscd(batch2)[0, :]
    return embedding.dot(embedding2).item()

def to_bchw(image):

    to_tensor = transforms.ToTensor()
    tensor = to_tensor(image)

    if len(tensor.shape) == 2:
        tensor = tensor.unsqueeze(0)

    if len(tensor.shape) == 3:
        tensor = tensor.unsqueeze(0)

    return tensor

def is_mostly_white(image, threshold=0.97, color=0.03, white=True):
    # Open the image and convert it to grayscale
    
    image_array = np.array(image)

    # Normalize pixel values to range 0-1 (0 = black, 1 = white)
    normalized_pixels = image_array / 255.0

    pixel_count = 0

    # Count white or pixels (close to 1.0)
    if not white:
        pixel_count = np.sum(normalized_pixels < 0.03)  # Adjust tolerance if needed
    else:
        pixel_count = np.sum(normalized_pixels > 0.97)
    
    total_pixel_count = normalized_pixels.size

    # Calculate the percentage of white pixels
    pixel_percentage = pixel_count / total_pixel_count
    print(pixel_percentage)

    # Return True if white pixels exceed the threshold
    return pixel_percentage > threshold


def compute(g_fold, no_segment, simple_name=False, sscd=False, ssim=False):

    model_name = "birefnet-general"
    session = new_session(model_name, providers=["CUDAExecutionProvider"])

    
    num_images = 500

    mem_folder = "memorized_images"
    gen_folder = "generated_images_nemo_cluster"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if ssim:
        ms_ssim = StructuralSimilarityIndexMeasure().to(device)
    else:
        ms_ssim = MultiScaleStructuralSimilarityIndexMeasure().to(device)


    orig_img = []
    generated_img = []
    mem_type = []
    scores = []
    background_scores = []
    foreground_scores = []

 
    for cluster in range(12):  
        for k in range(66):
                for j in range(0, 5):

                    q = PriorityQueue(maxsize=5)

                    if k < 10:
                        img_name = "{}_img_000{}_0{}.jpg".format(str(cluster), str(k), str(j))
                    elif k < 100:
                        img_name = "{}_img_00{}_0{}.jpg".format( str(cluster), str(k), str(j))
                    elif k < 1000:
                        img_name = "{}_img_0{}_0{}.jpg".format(str(cluster), str(k), str(j))
                    else:
                        img_name = "{}_img_{}_0{}.jpg".format( str(cluster), str(k), str(j))

                    if simple_name:
                        img_name = "{}_{}.jpg".format(str(cluster), str(k), str(j))

                    #print(img_name)


                    for i in range(0, 500):
                        if no_segment:
                            imname = "{}.jpg".format(str(i))
                        else:
                            imname = "{}.png".format(str(i))
                        print(imname)
                        memorized_path_background = os.path.join(mem_folder, "background", imname)
                        memorized_path_foreground = os.path.join(
                            mem_folder, "foreground", imname
                        ) 
                        im_path = os.path.join(mem_folder, imname)
                        
                        try:
                            if no_segment:
                                mem_image = to_bchw(Image.open(im_path).convert('RGBA').resize((512, 512), resample=Image.Resampling.LANCZOS))
                            else:
                                mem_background = to_bchw(Image.open(memorized_path_background).convert('RGBA').resize((512, 512), resample=Image.Resampling.LANCZOS))#.permute(0, 3, 1, 2)
                                mem_foreground = to_bchw(Image.open(memorized_path_foreground).convert('RGBA').resize((512, 512), resample=Image.Resampling.LANCZOS))#
                        

                        except Exception as e:
                            print(e)
                            continue
                        if no_segment:
                            mem_background = mem_image
                            mem_foreground = mem_image
                            gen_foreground = to_bchw(Image.open("generated_images_nemo_cluster" + '/' +img_name).convert('RGBA'))
                            gen_background = to_bchw(Image.open("generated_images_nemo_cluster" + '/' +img_name).convert('RGBA'))

                        else:
                            try:
                                if not no_segment:
                                    gen_foreground = to_bchw(Image.open("generated_images_nemo_cluster_foreground" + '/' +img_name).convert('RGBA'))
                                    gen_background = to_bchw(Image.open("generated_images_nemo_cluster_background" + '/' +img_name).convert('RGBA'))
                            except:
                            
                                try:
                                    gen_img = Image.open(gen_folder + "/" + img_name)
                                except:
                                    continue
                                gen_foreground, gen_background = fb_strip(session, gen_img)


                                gf = gen_foreground.convert('RGB')
                                gb = gen_background.convert('RGB')

                                gf.save("generated_images_nemo_cluster_foreground" + '/' +img_name)
                                gb.save("generated_images_nemo_cluster_background" + '/' +img_name)
                                gen_foreground = to_bchw(gen_foreground.convert('RGBA'))
                                gen_background = to_bchw(gen_background.convert('RGBA'))
                        
                        
                        if not sscd:
                            
                            gen_background_copy = Image.open("generated_images_nemo_cluster_background" + '/' +img_name).convert("L")
                            gen_foreground_copy = Image.open("generated_images_nemo_cluster_foreground" + '/' +img_name).convert("L")
                            if is_mostly_white(gen_background_copy):
                                gen_background = gen_foreground
                                print("Using only foreground.")
                            elif is_mostly_white(gen_foreground_copy, white=False):
                                gen_foreground = gen_background
                                print("Using only background.")

                            mem_background, mem_foreground = mem_background.to(device), mem_foreground.to(device)
                            gen_background, gen_foreground = gen_background.to(device), gen_foreground.to(device)

                            

                            fscore = ms_ssim(mem_foreground, gen_foreground).cpu().item()

                            if no_segment:
                                bscore = fscore
                            else:
                                bscore = ms_ssim(mem_background, gen_background).cpu().item()

                            score = (0.5 * fscore + 0.5 * bscore)

                            if not q.full():
                                q.put((score, bscore, fscore, imname, img_name))
                            else:
                                min_pair = q.get()
                                min_score = min_pair[0]
                                if score > min_score:
                                    q.put((score, bscore, fscore, imname, img_name))
                                else:
                                    q.put(min_pair)

                    print("-----------------------------------------")
                    print(q.qsize())
                    while not q.empty():
                        (score, bscore, fscore, imname, img_name) = q.get()
                        print(score)
                        print(imname)
                        print(img_name)
                        background_scores.append(bscore)
                        foreground_scores.append(fscore)
                        orig_img.append(imname)
                        generated_img.append(img_name)
                        scores.append(score)


                dict = {
                    "original image": orig_img,
                    "generated image": generated_img,
                    "Background Scores": background_scores,
                    "Foreground Scores": foreground_scores,
                    "Average Scores": scores,
                }
                df = pd.DataFrame(dict)
                if k == 0 and j == 0:
                    if no_segment:
                        df.to_csv(gen_folder + "/scores_no_seg.csv", index=False)
                    else:
                        df.to_csv(gen_folder + "/scores.csv", index=False)
                else:
                    if no_segment:
                        df.to_csv(gen_folder + "/scores_no_seg.csv", index=False, mode='a', header=False)
                    else:
                        df.to_csv(gen_folder + "/scores.csv", index=False, mode='a', header=False)
                orig_img = []
                generated_img = []
                mem_type = []
                scores = []
                background_scores = []
                foreground_scores = []
                
    return scores


def main(args):
    os.environ["OMP_NUM_THREADS"] = "2"
    scores = compute(args.sname, no_segment=args.no_segment, ssim=args.ssim)

    print(scores)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Add an argument
    parser.add_argument("--sname", type=str, required=False)
    parser.add_argument("--no_segment", action='store_true')
    parser.add_argument("--ssim", action='store_true')
    # Parse the argument
    args = parser.parse_args()
    main(args)
