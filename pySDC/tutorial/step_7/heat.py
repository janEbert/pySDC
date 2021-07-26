import matplotlib.pyplot as plt
import numpy as np


labels = ['RL+0', 'RL+RL', 'LU+0', 'LU+LU']
#mins = [9,14, 8]
#means = [11.6, 19.4, 11]
#maxima = [14, 25, 14]
mins = [9,10, 8]
means =     [23.0, 11.0, 35.0, 10.0]
maxima = [17.0, 12.0, 13.0, 8.0]

x = np.arange(len(labels))  # the label locations
width = 0.6  # the width of the bars

fig, ax = plt.subplots()
#rects1 = ax.bar(x - width/2, mins, width/2, label='nu = 1', color='green')
rects2 = ax.bar(x + width/2, maxima, width/2, label='nu=1', color='red')
rects3 = ax.bar(x, means, width/2, label='nu = 0.1', color='orange')

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
