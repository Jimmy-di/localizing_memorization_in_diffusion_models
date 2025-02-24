# import the necessary libraries 
import numpy as np 
import pandas as pd 
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.decomposition import PCA 
from sklearn.cluster import KMeans 
import matplotlib.pyplot as plt 
from utils.stable_diffusion import generate_images
from utils.stable_diffusion import load_sd_components, load_text_components
import torch
from sklearn.metrics import silhouette_score
# Dataset link:  
df = pd.read_csv("prompts/memorized_laion_prompts.csv", sep=';')
  
# Setup Tokenizer and Encoder
batch_size = 64
version = 'v1-4'
torch_device = "cuda"
    
tokenizer, text_encoder = load_text_components(version)
text_encoder.to(torch_device)

# Extract the sentence only 

tokens = []
embeddings = []
for index, row in df.iterrows():
    source_sentence = row['Caption']
    source_input = tokenizer([source_sentence],
                            padding="max_length",
                            max_length=tokenizer.model_max_length,
                            truncation=True,
                            return_tensors="pt")

    tokens.append(source_input.input_ids.cpu().data) 
    source_embedding = text_encoder(
    source_input.input_ids.to(text_encoder.device))[0].reshape(1, -1)
    embeddings.append(source_embedding.cpu().data.numpy()[0])

df['tokens'] = tokens
df['embeddings'] = embeddings
# cluster the documents using k-means 
max_i = 0
max_ss = 0

kmeans = KMeans(n_clusters=21,max_iter=500,random_state=1111111)
kmeans.fit_transform(embeddings) 
ss = silhouette_score(embeddings, kmeans.fit_predict(embeddings))
print(ss)

# Store data into a dataframe
results = pd.DataFrame() 
results['document'] = df.Caption
results['embeddings'] = df.embeddings
results['cluster'] = kmeans.labels_ 
results = results.rename(columns={"document": "Caption", "cluster":"cluster"})


result = pd.concat([results.set_index('Caption'), df.set_index('Caption')], axis=1, sort=True, join='inner')
result.drop(result.columns[result.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)

centroids  = kmeans.cluster_centers_  #means of shape [10,] 

print(centroids)

cent_dist = []

for i in range(len(centroids)):
    dist = []
    for j in range(len(centroids)):
        if i == j:
            dist.append(0)
        else: 
            distance = np.linalg.norm(centroids[i]-centroids[j])
            dist.append(distance)
    cent_dist.append(dist)

print(cent_dist)
   
'''
 # Test Dataset link:  
df_test = pd.read_csv("prompts/additional_laion_prompts.csv", sep=';')

tokens = []
embeddings = []
for index, row in df_test.iterrows():
    source_sentence = row['Caption']
    source_input = tokenizer([source_sentence],
                            padding="max_length",
                            max_length=tokenizer.model_max_length,
                            truncation=True,
                            return_tensors="pt")

    tokens.append(source_input.input_ids.cpu().data) 
    source_embedding = text_encoder(
    source_input.input_ids.to(text_encoder.device))[0].reshape(1, -1)
    embeddings.append(source_embedding.cpu().data.numpy()[0])


labels = kmeans.predict(embeddings)

df_test['cluster'] = labels
df_test.to_csv('prompts/additional_laion_prompts_cluster_all_embeddings.csv')

df_test.drop(df_test.columns[df_test.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)
df_test.sort_values(['cluster'])


for i in range(12):
    
    df_test_i = df_test.loc[df_test['cluster'] == i]
    if df_test_i.shape[0] > 1000:
        df_test_i = df_test.iloc[:1000] 
    df_test_i.to_csv('prompts/additional_laion_prompts_cluster_{}_embeddings.csv'.format(i), sep=";")
    
print(df_test.head()) 
'''

# create a dataframe to store the results 

'''
for i in range(21):
    
    result_i = result.loc[result['cluster'] == i]
    result_i = result_i.drop(['embeddings', 'tokens'], axis=1).sort_values(['cluster'])
    print(result_i.head(5))
    result_i.to_csv('prompts/memorized_laion_prompts_cluster_{}_embeddings.csv'.format(i), sep=";")

result = result.drop(['embeddings', 'tokens'], axis=1).sort_values(['cluster'])
print(result.head())

result.to_csv('prompts/memorized_laion_prompts_cluster_all_embeddings.csv', sep=";")

#result = result.sort_values(by=['cluster'])
'''



# plot the results 
#for i in range(num_clusters): 12
#    plt.scatter(reduced_data[kmeans.labels_ == i, 0], 
#                reduced_data[kmeans.labels_ == i, 1],  
#                s=10,  
#                label="Cluster {}".format(i)) 
#plt.legend() 
#plt.savefig("cluster")

