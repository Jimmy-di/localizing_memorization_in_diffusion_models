import numpy as np
import matplotlib.pyplot as plt
import os
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import glob
import os
import torch
import argparse
import heapq

def create_parser():
    parser = argparse.ArgumentParser(description='Graph Results')

    parser.add_argument(
        '-c',
        '--cluster',
        default=0, 
        type=int,
        dest="cluster",
        help='Cluster Number\').'
    )
    parser.add_argument(
        '-t',
        '--threshold',
        default='0.400', 
        type=str,
        dest="threshold",
        help='Threshold\').'
    )

    args = parser.parse_args()
    return args

def main():

    args = create_parser()
    # Load the OpenAI CLIP Model
    print('Loading CLIP Model...')
    model = SentenceTransformer('clip-ViT-B-32')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    in_cluster_ssim = []
    out_cluster_ssim = []
    add_cluster_ssim = []

    capacity = 200
    out_cluster_names = []
    in_cluster_names = []

    max_i1 = ""
    max_i2 = ""
    out_cluster_max = 0

    train_folder = "training_imgs"
    
    foldername = "generated_images_orig_blocked_r_cluster_"+ str(args.cluster)+"_embeddings_block_all_" +str(args.threshold)
    extra_folder = "add_generated_images_orig_blocked_r_cluster_"+ str(args.cluster)+"_embeddings_block_all_" +str(args.threshold)

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
            
            if cluster == str(args.cluster):
                in_cluster_ssim.append((processed_images[0], image1, image2))

                if len(in_cluster_names) < capacity:
                    heapq.heappush(in_cluster_names, (processed_images[0], image1, image2))
                else:
                    spilled_value = heapq.heappushpop(in_cluster_names, (processed_images[0], image1, image2))
            else:
                out_cluster_ssim.append((processed_images[0], image1, image2))
                if len(out_cluster_names) < capacity:
                    heapq.heappush(out_cluster_names, (processed_images[0], image1, image2))
                else:
                    spilled_value = heapq.heappushpop(out_cluster_names, (processed_images[0], image1, image2))
                if processed_images[0] > out_cluster_max:
                    max_i1 = image1
                    max_i2 = image2
                    out_cluster_max = processed_images[0]
                    
    # for image1 in os.listdir(extra_folder):

    #     for image2 in os.listdir(train_folder):
    #         image_names = []
    #         if image1.endswith(".jpg"):
    #             image_names.append(os.path.join(extra_folder, image1))
    #         else:
    #             continue
    #         cluster = image2.strip(".jpg").split("_")[0]
    #         print(cluster)
    #         image_names.append(os.path.join(train_folder, image2))
    #         encoded_image = model.encode([Image.open(filepath) for filepath in image_names], batch_size=2, convert_to_tensor=True, show_progress_bar=True)

    #         # Now we run the clustering algorithm. This function compares images aganist 
    #         # all other images and returns a list with the pairs that have the highest 
    #         # cosine similarity score
    #         processed_images = util.paraphrase_mining_embeddings(encoded_image)[0]
            
    #         add_cluster_ssim.append(processed_images[0])


    if len(out_cluster_ssim) > 0:
        #print(sum(out_cluster_ssim) / len(out_cluster_ssim))
        print(max_i1)
        print(max_i2)
        print(out_cluster_max)

    with open('outclus_blocked_{}_{}.out'.format(args.cluster, args.threshold), 'w') as fp:
        fp.write('\n'.join('{} {} {}'.format(x[0],x[1], x[2]) for x in out_cluster_ssim))

    with open('inclus_blocked_{}_{}.out'.format(args.cluster, args.threshold), 'w') as fp:
        fp.write('\n'.join('{} {} {}'.format(x[0],x[1], x[2]) for x in in_cluster_ssim))

    np.savetxt("addclus_blocked_{}_{}.out".format(args.cluster, args.threshold), np.array(add_cluster_ssim))
    
    with open('outclus_blocked_{}_{}_imgs.out'.format(args.cluster, args.threshold), 'w') as fp:
        fp.write('\n'.join('{} {} {}'.format(x[0],x[1], x[2]) for x in out_cluster_names))
    #print(out_cluster_ssim)       
    with open('inclus_blocked_{}_{}_imgs.out'.format(args.cluster, args.threshold), 'w') as fp:
        fp.write('\n'.join('{} {} {}'.format(x[0],x[1], x[2]) for x in in_cluster_names))
 
    print("In-cluster score:")
    if len(in_cluster_ssim) > 0:
        print(sum(in_cluster_ssim) / len(in_cluster_ssim))
    #print((in_cluster_ssim))
    
"""     print("Additional cluster score:")
    if len(add_cluster_ssim) > 0:
        print(sum(add_cluster_ssim) / len(add_cluster_ssim)) """


if __name__ == "__main__":
    main()

            
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


