import matplotlib.pyplot as plt
import numpy as np


labels = ['RL+0','RL+RL', 'Opt+0','Opt+Opt', 'LU+0', 'LU+LU']

mins =  [11, 8, 15, 14, 12, 11]
means =  [13.5, 9.0, 18.0, 17.5, 14.5, 13.2]
maxima =  [16, 10, 21, 21, 17, 16]


x = np.arange(len(labels))  # the label locations
width = 0.6  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, mins, width/2, label='min', color='green')
rects2 = ax.bar(x, means, width/2, label='mean', color='orange')
rects3 = ax.bar(x + width/2, maxima, width/2, label='max', color='red')

rects1[0].set_color('lightblue')
rects1[1].set_color('lightblue')
rects1[2].set_color('lightgreen')
rects1[3].set_color('lightgreen')
rects1[4].set_color('tomato')
rects1[5].set_color('tomato')
#rects1[3].set_color('lightblue')
#rects1[4].set_color('lightgreen')
#rects1[5].set_color('tomato')

rects2[0].set_color('blue')
rects2[1].set_color('blue')
rects2[2].set_color('green')
rects2[3].set_color('green')
rects2[4].set_color('red')
rects2[5].set_color('red')
#rects2[3].set_color('blue')
#rects2[4].set_color('green')
#rects2[5].set_color('red')

rects3[0].set_color('darkblue')
rects3[1].set_color('darkblue')
rects3[2].set_color('darkgreen')
rects3[3].set_color('darkgreen')
rects3[4].set_color('darkred')
rects3[5].set_color('darkred')
#rects3[3].set_color('darkblue')
#rects3[4].set_color('darkgreen')
#rects3[5].set_color('darkred')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('#iterations', fontsize=30)
ax.yaxis.set_tick_params(labelsize=20)
#ax.set_title('Scores by group and gender')
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=30)
ax.legend(fontsize=12)
#ax.tick_params(axis='both', which='minor', labelsize=20)


ax.bar_label(rects1, padding=0, fontsize=20)
ax.bar_label(rects2, padding=0, fontsize=20)
ax.bar_label(rects3, padding=0, fontsize=20)

fig.tight_layout()

plt.show()
