import matplotlib.pyplot as plt
import numpy as np


labels = ['RL', 'Opt', 'LU']

mins =  [9, 12, 8]
means =  [11.0, 15.4, 10.8]
maxima =  [13, 19, 13]



x = np.arange(len(labels))  # the label locations
width = 0.6  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, mins, width/2, label='min #iterations')
rects2 = ax.bar(x, means, width/2, label='mean #iterations')
rects3 = ax.bar(x + width/2, maxima, width/2, label='max #iterations' )

rects1[0].set_color('lightblue')
rects1[1].set_color('lightgreen')
rects1[2].set_color('tomato')

rects2[0].set_color('blue')
rects2[1].set_color('green')
rects2[2].set_color('red')

rects3[0].set_color('darkblue')
rects3[1].set_color('darkgreen')
rects3[2].set_color('darkred')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('#iterations', fontsize=40)
ax.yaxis.set_tick_params(labelsize=30)
#ax.set_title('Scores by group and gender')
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=40)
ax.legend(fontsize=25, loc='upper left')
ax.tick_params(axis='both', which='minor', labelsize=40)




ax.bar_label(rects1, padding=0, fontsize=30)
ax.bar_label(rects2, padding=0, fontsize=30)
ax.bar_label(rects3, padding=0, fontsize=30)

fig.tight_layout()

plt.show()
