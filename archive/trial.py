from PIL import Image
import pandas as pd
import requests
import time
import ast


total = 0
num_vm = 0
num_tm = 0
list_count = []

#for i in range(11):
#    df = pd.read_csv("results/memorization_statistics_r_cluster_{}_embeddings_0428_v1_4.csv".format(str(i)), sep=";")
#    count = 0
#    for index, row in df.iterrows():

#        x = ast.literal_eval(row['Refined Neurons'])
#        #print(type(x))
#        for lst in x:
        #print(type(lst_str))
        #lst = ast.literal_eval(lst_str)
#            count+=len(lst)

#    print(count)
#    list_count.append(count)
#print(sum(list_count)/len(list_count))




#for i in range(11):
    

for i in range(12):
    df = pd.read_csv("generated_images_orig_blocked_r_cluster_{}_embeddings_block_all_0428/scores.csv".format(str(i)), index_col=False)
    df.drop_duplicates()
    for index, row in df.iterrows():
        if row['score'] >= 0.8:
            df.at[index, 'conclusion'] = 'VM'
        elif row['score'] > 0.51:
            df.at[index, 'conclusion'] = 'TM'
        else:
            df.at[index, 'conclusion'] = 'NM'
    count = df.groupby('conclusion').count()
    for index, row in count.iterrows():
        print(row)
        total+=int(row['original image'])
        if row.name == 'NM':
            continue
        elif row.name == 'TM':
            num_tm += int(row['original image'])
        else:
            num_vm += int(row['original image'])

print(num_tm/(num_vm+num_tm+total))
print(num_vm/total)