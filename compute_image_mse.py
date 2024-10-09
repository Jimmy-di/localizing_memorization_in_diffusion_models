import numpy as np
import matplotlib.pyplot as plt
import os
import skimage as ski
from skimage import data, img_as_float
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error
from skimage.util import crop

if __name__ == '__main__':
    mse_results = []
    ssim_results = []

    images_emma = []
    images_george = []

    emma_train = ski.io.imread("emmawatsonlead.jpg")[1:513, 1:513]
    george_train = ski.io.imread("GeorgeMartin.jpg")[1:513, 1:513]

    foldername = "generated_images_blocked_few_0.35_2"

    print(emma_train.shape)

    for i in range(5):
        for j in range(5):
            image_name = "img_000" + str(i) + "_0" + str(j) + ".jpg"
            filename = os.path.join(foldername, image_name)
            img = ski.io.imread(filename)
            images_george.append(img)
            #print(img.shape)
    
    for img in images_george:
        mse_none = mean_squared_error(img, george_train)
        b_max = max(img.max(), george_train.max())
        b_min = min(img.min(), george_train.min())
        ssim_none = ssim(img, george_train, channel_axis=-1, data_range=255)#, data_range=b_max - b_min)
        mse_results.append(mse_none)
        ssim_results.append(ssim_none)

    print("MSE Score:")
    print(mse_results)
    print("SSIM Scores:")
    print(ssim_results)