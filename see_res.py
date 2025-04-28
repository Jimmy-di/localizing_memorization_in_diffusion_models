import pandas as pd

df = pd.read_csv("./generated_images_orig_unblocked_v2_1_additional_prompts_faiss_20/report.csv")
#df = df.sort_values(by=['image name', 'segment score'])
df = df.sort_values(by=['segment score'])

print(df.head(20))
df = df.groupby(['image name']).max()

print (df.describe())
print(df.head(5))
#print(df["score"].mean())