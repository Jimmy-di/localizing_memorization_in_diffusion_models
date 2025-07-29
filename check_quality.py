
from torchvision import transforms
from PIL import Image
import matplotlib.ticker as mtick
import os
import torch
import pyiqa
from pyiqa import create_metric
import numpy as np
torch.cuda.empty_cache()
sfid = create_metric('sfid')#.cuda()
#quality_score = qalign(input, task_='quality')
#aesthetic_score = qalign(input, task_='aesthetic')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

folders = ['generated_images_orig_unblocked_v1_4_1',
            'generated_images_orig_blocked_v1_4_1',
            'generated_images_additional_prompts_500_1',
            'generated_images_mitigated',
            'generated_images_blocked_cp',
#            'generated_images_nemo_cluster'
            ]

folder = 'generated_images_nemo_cluster'

score = []
#for l in range(12):
for folder in folders:
    for k in range(500):
        for j in range(5):

            img_name = "img_{}_{}.jpg".format(str(k).zfill(4), str(j).zfill(2))

            if folder == 'generated_images_mitigated':
                img_name = "{}_{}.jpg".format(str(k), str(j))
            elif folder == 'generated_images_blocked_cp':
                img_name = "img_{}_{}.jpg".format(str(k), str(j))

            img_path = os.path.join(folder, img_name)
            try:

                img = Image.open(img_path).convert('RGB')

            except FileNotFoundError:
                print(f"File not found: {img_path}")
                continue
            score_fr = sfid('img_path', './coco2017')
            score.append(score_fr.item())
            #scores.append(score)

print(f"Scores for {folder}: {score}")
print("")
print(f"Average score for {folder}: {np.mean(score)}")
