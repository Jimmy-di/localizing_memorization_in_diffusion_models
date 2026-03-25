import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import matplotlib.ticker as mtick

df = pd.read_csv("generated_images_orig_unblocked_v1_4_1/scores.csv")
df2 = pd.read_csv("generated_images_orig_unblocked_v1_4_1_50_2/score.csv")

df = df.sort_values("Average Scores").groupby("generated image").last().reset_index()
df2 = df2.sort_values("Average Scores").groupby("generated image").last().reset_index()

benchmark = []
cat_name = ["VM", 'BM', "FM", "NM"]


for index, row in df.iterrows():
    if row['Background Scores'] >= 0.8 and row['Foreground Scores'] >= 0.8:
        benchmark.append("VM")
    elif row['Background Scores'] >= 0.8:
        benchmark.append("BM")
    elif row['Foreground Scores'] >= 0.8:
        benchmark.append("FM")
    else:
        benchmark.append("NM")
    
score = 0
index = 0
for index, row in df2.iterrows():
    if benchmark[index] == 'VM':
        if row['Background Scores'] >= 0.8 and row['Foreground Scores'] >= 0.8:
            score += 0
        elif row['Background Scores'] >= 0.8:
            score+=1.5
        elif row['Foreground Scores'] >= 0.8:
            score+=0.5
        else:
            score+=2
    elif benchmark[index] == 'BM':
        if row['Background Scores'] >= 0.8 and row['Foreground Scores'] >= 0.8:
            score -= 1.5
        elif row['Background Scores'] >= 0.8:
            score+=0
        elif row['Foreground Scores'] >= 0.8:
            score-=0.5
        else:
            score+=0.5
    elif benchmark[index] == 'FM':
        if row['Background Scores'] >= 0.8 and row['Foreground Scores'] >= 0.8:
            score -= 0.5
        elif row['Background Scores'] >= 0.8:
            score+=1
        elif row['Foreground Scores'] >= 0.8:
            score+=0
        else:
            score+=1.5
    else:
        if row['Background Scores'] >= 0.8 and row['Foreground Scores'] >= 0.8:
            score -= 2
        elif row['Background Scores'] >= 0.8:
            score-=0.5
        elif row['Foreground Scores'] >= 0.8:
            score-=1.5
        else:
            score+=0
    index+=1

print(score)

scores = [1844, 1676.5, 1999.5]
costs = [10, 15, 12]
cost2 = [ -2.3, 16.8, 0.5]
labels = ["NeMo", "Detect Mem", "Wanda"]

fig, ax = plt.subplots()
ax.scatter(cost2, scores)

# Annotate each point
for i, txt in enumerate(labels):
    ax.annotate(txt, (cost2[i], scores[i]-8), textcoords="offset points", xytext=(0,10), ha='center')

plt.ylabel("Mitigation Score")
plt.xlabel("Utility Loss (%)")
plt.title("Mitigation Score vs Image Quality Trade-off")
plt.savefig("./myPlot15.png")
# blocked 1844
# wen 1676.5
# wanda 1999.5