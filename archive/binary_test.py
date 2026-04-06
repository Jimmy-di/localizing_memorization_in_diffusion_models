import torch
import clip
from PIL import Image
import os
import numpy as np
import json

import PIL
from PIL import Image

import pandas as pd
from queue import PriorityQueue
import math
from sklearn.metrics import f1_score, roc_auc_score, roc_curve
from operator import xor
from sklearn.metrics import roc_curve

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

mem_status = []

vm_index = []
tm_index = []


def accuracy(y_true, y_pred):
    """
    Calculates the accuracy between two lists.

    Args:
        y_true: A list of true values.
        y_pred: A list of predicted values.

    Returns:
        The accuracy as a float.
    """
    correct_predictions = 0
    for i in range(len(y_pred)):
        if y_pred[i] == y_true[i]:
            correct_predictions += 1
    return correct_predictions / len(y_true)


y_true = []
y_score = []
y_pred = []
gt = []

df1 = pd.read_csv("generated_images_orig_unblocked_v1_4_1/scores_no_seg_ssim.csv")
df1 = df1.groupby(by=['generated image']).max("Average Scores")



df = pd.read_csv("generated_images_orig_unblocked_v1_4_1/scores_grouped.csv")
gt = df['Mem Type'].to_list()
if (len(gt) > len(df1)):
    print(len(gt), len(df1))
    gt = gt[:1675]

df1['Mem Type'] = gt


#df1 = pd.read_csv("generated_images_orig_unblocked_v1_4_1/scores_no_seg_sscd.csv")
#df1 = df1.groupby(by=['generated image']).max("Average Scores")

print(df1.head())
df2 = pd.read_csv("generated_images_additional_prompts_500_1/scores_no_seg_ssim.csv").groupby(by=['generated image']).max("Average Scores")


for index, row in df1.iterrows():
    if row['Mem Type'] =='VM': #or row['Mem Type'] =='VM':
        y_true.append(1)
        y_score.append(row["Average Scores"])

        #if xor(row["Background Scores"] >= 0.80, row["Foreground Scores"] >= 0.80):
        if row["Average Scores"] >= 0.80:# and row["Average Scores"] < 0.80:
            y_pred.append(1)
        else:
            y_pred.append(0)

    elif row['Mem Type'] =='TM': #or row['Mem Type'] =='TM':
        y_true.append(0)
        y_score.append(row["Average Scores"])

        if (row["Average Scores"] >= 0.60 and row["Average Scores"] < 0.80):
            y_pred.append(1)
        else:
            y_pred.append(0)


print(len(y_true))
k = 0

for index, row in df2.iterrows():
    break
    y_true.append(0)
    y_score.append(row["Average Scores"])

    if row["Average Scores"] >= 0.60 and row["Average Scores"] < 0.80:
    #if xor(row["Background Scores"] >= 0.80, row["Foreground Scores"] >= 0.80):
        y_pred.append(1)
    else:
        y_pred.append(0)
    if k == 1500:
        break
    k+=1


print(roc_auc_score(y_true, y_score))
fpr, tpr, thresholds = roc_curve(y_true, y_score)

print(f1_score(y_true, y_pred))
closest_fpr_index = np.argmin(np.abs(fpr - 0.01))

# Get the TPR at that index
tpr_at_1percent_fpr = tpr[closest_fpr_index]
print(f"TPR at 1% FPR: {tpr_at_1percent_fpr}")

print(accuracy(y_true, y_pred))

    




