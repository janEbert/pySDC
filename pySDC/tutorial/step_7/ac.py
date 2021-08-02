import matplotlib.pyplot as plt
import numpy as np


labels = ['RL', 'Opt', 'LU']

means =  [18.76, 18.79, 23.63] #0.04
maxima =  [5.76, 6.71, 5.87] #1

x = np.arange(len(labels))  # the label locations
width = 0.6  # the width of the bars

fig, ax = plt.subplots()
#rects1 = ax.bar(x - width/2, mins, width/2, label='min', color='green')
rects2 = ax.bar(x, means, width/2, label='eps = 0.04', color='orange')
rects3 = ax.bar(x + width/2, maxima, width/2, label='eps = 1', color='red')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('#iterations', fontsize=20)
ax.yaxis.set_tick_params(labelsize=12)
#ax.set_title('Scores by group and gender')
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=20)
ax.legend(fontsize=12)
#ax.tick_params(axis='both', which='minor', labelsize=20)


#ax.bar_label(rects1, padding=0, fontsize=12)
ax.bar_label(rects2, padding=0, fontsize=12)
ax.bar_label(rects3, padding=0, fontsize=12)

fig.tight_layout()

plt.show()
