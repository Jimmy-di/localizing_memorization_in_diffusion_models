import numpy as np
import matplotlib.pyplot as plt
import os
import skimage as ski

from sentence_transformers import SentenceTransformer, util
from PIL import Image
import glob
import os
import torch

if __name__ == '__main__':

    # Load the OpenAI CLIP Model
    print('Loading CLIP Model...')
    model = SentenceTransformer('clip-ViT-B-32')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    in_cluster_ssim = []
    out_cluster_ssim = []

    max_i1 = ""
    max_i2 = ""
    out_cluster_max = 0

    train_folder = "training_imgs"
    
    foldername = "generated_images_orig_blocked_r_cluster_2_embeddings_block_all_0.428"

    for image1 in os.listdir(foldername):

        for image2 in os.listdir(train_folder):
            image_names = []
            if image1.endswith(".jpg"):
                image_names.append(os.path.join(foldername, image1))
            else:
                continue
            cluster = image2.strip(".jpg").split("_")[0]
            print(cluster)
            image_names.append(os.path.join(train_folder, image2))
            encoded_image = model.encode([Image.open(filepath) for filepath in image_names], batch_size=2, convert_to_tensor=True, show_progress_bar=True)

            # Now we run the clustering algorithm. This function compares images aganist 
            # all other images and returns a list with the pairs that have the highest 
            # cosine similarity score
            processed_images = util.paraphrase_mining_embeddings(encoded_image)[0]
            
            if cluster == '2':
                in_cluster_ssim.append(processed_images[0])
            else:
                out_cluster_ssim.append(processed_images[0])
                if processed_images[0] > out_cluster_max:
                    max_i1 = image1
                    max_i2 = image2
                    out_cluster_max = processed_images[0]

    print("Out-cluster score:")
    if len(out_cluster_ssim) > 0:
        #print(sum(out_cluster_ssim) / len(out_cluster_ssim))
        print(max_i1)
        print(max_i2)
        print(out_cluster_max)


    np.savetxt("outclus.out", np.array(out_cluster_ssim))
    np.savetxt("inclus.out", np.array(in_cluster_ssim))
    #print(out_cluster_ssim)       
 
    print("In-cluster score:")
    if len(in_cluster_ssim) > 0:
        print(sum(in_cluster_ssim) / len(in_cluster_ssim))
    #print((in_cluster_ssim))


            
    




""" 
    for i in range(43):
        if i == 1 or i == 4:
            continue
        train_imgname = str(i)+".jpg"
        train_imgname = os.path.join(train_folder, train_imgname)
        original_train = ski.io.imread(train_imgname)
        if (original_train.shape[0] < 512):
            continue
        else:
            original_train = original_train#[1:513, 1:513]
        training_imgs.append(original_train)
        #print(original_train.shape)
        for j in range(5):
            if i >= 10:
                image_name = "img_00" + str(i) + "_0" + str(j) + ".jpg"
            else:
                image_name = "img_000" + str(i) + "_0" + str(j) + ".jpg"
            
            filename = os.path.join(foldername, image_name)
            img = ski.io.imread(filename)
            img = resize(img, original_train.shape)
            images_test.append(img)
            #print(img.shape)
    
    for index, img in enumerate(images_test):
        original_train = training_imgs[index]
        mse_none = mean_squared_error(img, original_train)
        b_max = max(img.max(), original_train.max())
        b_min = min(img.min(), original_train.min())
        ssim_none = ssim(img, original_train, channel_axis=-1,  data_range=b_max - b_min)#data_range=255,
        mse_results.append(mse_none)
        ssim_results.append(ssim_none)

    print("MSE Score:")
    print(mse_results)
    print("SSIM Scores:")
    print(ssim_results)
    print(sum(ssim_results) / len(ssim_results)) """

    # 0.015920809296883415 0 0.20901010856038457
    # 0.016040029611762294 1 0.2227764844700998
    # 0.016060485182275413 2 0.2526291204852843
    # 0.016113187854269052 3 0.25762852069593045
    # 0.01625903007479146 4 0.260702489771103
    # 0.016354048481182153 5 0.2648800816307255
    # 0.01644456231277762 6 0.2567549127369116
    # 0.01650309045448384 7 0.256509082199462
    # 0.016634469295635116 8 0.2673893547250485
    # 0.01686139260124685 9 0.2776262103225899
    # 0.016905791348479898 10 0.27648891651980745
    # 0.01709759700059836 15 0.2628697567929559
    # 0.016791083869611374 20 0.2507286482032497


