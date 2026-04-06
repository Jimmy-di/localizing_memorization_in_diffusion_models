import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def add_labels(x, y, labels):
    for i in range(len(x)):
        plt.text(x[i], y[i], labels[i]) 

#x = np.array([0, 0.123540991, 0.146432604, 0.165885033, ])
x = np.array([0, 0.154824829, 0.176494732, 0.232498047 ])
y = np.array([4.027576, 3.628532, 3.41662, 3.552897])
yerr = np.array([0.522811, 0.775095, 0.547919, 0.745581]) # Error in y-direction

labels = ['Original', 'NeMo', 'Wen et. al.', 'Wanda']
add_labels(x, y, labels)

plt.errorbar(x, y, yerr=yerr, fmt='o', capsize=5, ecolor='red')


plt.xlabel('Average Change in Foreground SIM Score')
plt.ylabel('Quality Score (QAlign)')
plt.title('Image Quality Scores for Mitigation Methods')
plt.grid(True)
plt.savefig('quality_scores_with_error_bars_foreground.png', dpi=300)