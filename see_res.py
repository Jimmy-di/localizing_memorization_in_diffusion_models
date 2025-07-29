import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, f1_score

#df = pd.read_csv("prompts/additional_laion_prompts.csv", sep=';')
#df = df.sample(n=500, random_state=1)
#df.to_csv('prompts/additional_laion_prompts_500.csv', index=False, sep=";")

df = pd.read_csv("generated_images_additional_prompts_500_1/scores_no_seg.csv")
df2 = pd.read_csv("generated_images_additional_prompts_500_1/scores.csv")

#df = df.sort_values("Average Scores").groupby("generated image").last().reset_index()
df2 = df2.sort_values("Average Scores").groupby("generated image").last().reset_index()


df['original image'] = df['original image'].str.split('.').str[0]
df2['original image'] = df2['original image'].str.split('.').str[0]

print(df.head())
print(df2.head())

#for row_number, row in df2.iterrows():
#    if row["Average Scores"] >= 0.59:
#        y_score.append(row["Average Scores"])
#        y_true.append(1)
#    else:
#        rem_count+=1


#for row_number, row in df.iterrows():
#    y_score.append(row["Average Scores"])
#    y_true.append(0)

#roc_auc = roc_auc_score(y_true, y_score)
#print(roc_auc)
#optimal_threshold = thresholds[np.argmax(f1_scores)]

count_vm = [0, 0, 0, 0, 0]
count_bm = [0, 0, 0, 0, 0]
count_fm = [0, 0, 0, 0, 0]
count_nm = [0, 0, 0, 0, 0]
rem_count = 0

s = set()
record = [0,0,0,0]
for row_number, row in df2.iterrows():
    if row_number % 5 == 0 and len(s) > 0:
        num_gen = len(s) - 1
        count_vm[num_gen] += record[0]
        count_bm[num_gen] += record[1]
        count_fm[num_gen] += record[2]
        count_nm[num_gen] += record[3]
        record = [0,0,0,0]
        s = set()
    
    s.add(row['original image'])

    second_row = df[(df['generated image'] == row['generated image']) & (df['original image'] == row['original image'])]
    #print(second_row['Average Scores'])
    if row['Background Scores'] >= 0.8 and row['Foreground Scores'] >= 0.8:
        record[0] += 1
    elif row['Background Scores'] >= 0.8:
        record[1] += 1
    elif row['Foreground Scores'] >= 0.8:
        record[2] += 1
    else:
        record[3] += 1
    
    rem_count+=1

print(sum(count_vm) / rem_count)
print(sum(count_bm)/ rem_count)
print(sum(count_fm)/ rem_count)
print(sum(count_nm)/ rem_count)

# total_sum = np.array(count_vm) + np.array(count_bm) + np.array(count_fm) + np.array(count_nm)
# percentages = [ (v / 2500) * 100 for v in total_sum ]

# print(percentages)
# print(total_sum)



# #bins = np.linspace(0, 1.0, 100)
# fig, ax = plt.subplots()

# p = plt.bar([1, 2, 3, 4, 5], count_vm, label='VM')
# ax.bar_label(p, label_type='center')

# p = plt.bar([1, 2, 3, 4, 5], count_bm, label='BM', bottom=count_vm)
# ax.bar_label(p, label_type='center')
# p = plt.bar([1, 2, 3, 4, 5], count_fm, label='FM', bottom=np.array(count_vm) + np.array(count_bm))
# ax.bar_label(p, label_type='center')

# p = plt.bar([1, 2, 3, 4, 5], count_nm, label='NM', bottom=np.array(count_vm) + np.array(count_bm) + np.array(count_fm))
# ax.bar_label(p, label_type='center')

# plt.xlabel('# of Copied Images in 5 Generations')
# plt.ylabel('# of Images')
# plt.title('# of Copied Images Detected in 5 Generations')
# plt.legend()

# total_sum = np.array(count_vm) + np.array(count_bm) + np.array(count_fm) + np.array(count_nm)
# percentages = [ (v / 2500) * 100 for v in total_sum ]

# print(percentages)
# print(total_sum)
# for i, p in enumerate(percentages):
#     plt.text(i+1, total_sum[i] + 40, f'{p:.1f}%', ha='center', va='top') 
#     # Adjust 'values[i] + 1' for vertical position, 'ha' for horizontal alignment

# plt.savefig("my_plot13.png")

# # plt.hist(df['Average Scores'].tolist(), bins, alpha=0.5, label='Non Memorized Images')
# # plt.hist(df2['Average Scores'].tolist(), bins, alpha=0.5, label='Memorized Images')

# # plt.title('Comparison of Non Memorized vs Memorized Images using BF-MSSSIM')
# # plt.xlabel('BF-SSIM')
# # plt.ylabel('Frequency')
# # plt.legend()
# #plt.savefig("my_plot4.png")