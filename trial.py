from PIL import Image
import pandas as pd
import requests
import time

df = pd.read_csv("prompts/memorized_laion_prompts.csv", sep=';')
urls = df['URL'].to_list()
print(urls)

SSCD_scores = []
clip_scores = []
segment_scores = []

for i, url in enumerate(urls):
    try:
        im = Image.open(requests.get(url, stream=True, timeout=2).raw)
        time.sleep(2)
    except:
        continue

    im = im.convert('RGB').resize((512, 512), Image.Resampling.LANCZOS)
    im.save("memorized_images/{}.jpg".format(str(i)))

