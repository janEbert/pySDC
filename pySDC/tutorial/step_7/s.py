import matplotlib.pyplot as plt
import numpy as np


labels = ['RL', 'Opt', 'LU']



spread =  [12.25, 14.4, 11.7] #[12.4, 14.4, 11.7]
zero   =  [13.55, 17.4, 11.7]  #[13.65, 17.4, 11.7]

x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects2 = ax.bar(x - width/2, spread, width/2,  label='spread', color='green')
#rects2 = ax.bar(x, spread, width/2, label='spread', color='orange')
rects3 = ax.bar(x + width/2, zero, width/2, label='zero',  color='red')

#rects3[0].set_color('white')
#rects3[1].set_color('white')
#rects3[2].set_color('white')

rects2[0].set_color('blue')
rects2[1].set_color('green')
rects2[2].set_color('red')

rects3[0].set_color('darkblue')
rects3[1].set_color('darkgreen')
rects3[2].set_color('darkred')


# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('#iterations', fontsize=40)
ax.yaxis.set_tick_params(labelsize=20)
#ax.set_title('Scores by group and gender')
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=40)
ax.legend(fontsize=20)
#ax.tick_params(axis='both', which='minor', labelsize=20)


#ax.bar_label(rects1, padding=0, fontsize=12)
ax.bar_label(rects2, padding=0, fontsize=30)
ax.bar_label(rects3, padding=0, fontsize=30)

fig.tight_layout()

plt.show()
