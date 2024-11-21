# import the necessary libraries 
import numpy as np 
import pandas as pd 
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.decomposition import PCA 
from sklearn.cluster import KMeans 
import matplotlib.pyplot as plt 
  
# Dataset link:  
df = pd.read_csv("prompts/memorized_laion_prompts.csv", sep=';')
  
# Extract the sentence only 
sentence = df.Caption
  
# create vectorizer 
vectorizer = TfidfVectorizer(stop_words='english') 
  
# vectorizer the text documents 
vectorized_documents = vectorizer.fit_transform(sentence) 
  
# reduce the dimensionality of the data using PCA 
pca = PCA(n_components=2) 
reduced_data = pca.fit_transform(vectorized_documents.toarray()) 
  
  
# cluster the documents using k-means 
num_clusters = 10
kmeans = KMeans(n_clusters=num_clusters, n_init=5, 
                max_iter=500, random_state=42) 
kmeans.fit(vectorized_documents) 
  
  
# create a dataframe to store the results 
results = pd.DataFrame() 
results['document'] = sentence 
results['cluster'] = kmeans.labels_ 
results = results.rename(columns={"document": "Caption", "cluster":"cluster"})




result = pd.concat([results.set_index('Caption'), df.set_index('Caption')], axis=1, sort=True, join='inner')
result.drop(result.columns[result.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)
result = result.loc[result['cluster'] == 8]
print(result.head(10))


result = result.sort_values(by=['cluster'])
result = result.drop(['cluster'], axis=1)


result.to_csv('prompts/memorized_laion_prompts_clustered_8.csv', sep=";")
# plot the results 
#for i in range(num_clusters): 
#    plt.scatter(reduced_data[kmeans.labels_ == i, 0], 
#                reduced_data[kmeans.labels_ == i, 1],  
#                s=10,  
#                label="Cluster {}".format(i)) 
plt.legend() 
plt.savefig("cluster")