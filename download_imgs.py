import pandas as pd   
import wget
import time  # for sleep
folder = 'training_imgs'
for i in range(21):
   df = pd.read_csv("prompts/memorized_laion_prompts_cluster_{}_embeddings.csv".format(i), sep=';')
   

   url = df.URL.tolist()
   if len(url)> 500:
      url = url[:500]
   #clusters = df.cluster.tolist()
   for j, photos in enumerate(url):
      try: 
         print('\nGet:', photos)   
         wget.download(photos, folder + '/{}_{}.jpg'.format(i, j))
         time.sleep(1)  # pause 1 second, if needed
      except Exception as ex:
         print('Failed to get:',photos, ex)