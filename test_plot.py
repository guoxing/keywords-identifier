import numpy as np
import matplotlib.pyplot as plt

alphab = ['A', 'B', 'C', 'D', 'E', 'F']
frequencies = [23, 44, 12, 11, 2, 10]

pos = np.arange(len(alphab))
width = 1.0     # gives histogram aspect to the bar diagram

ax = plt.axes()
ax.set_xticks(pos + (width / 2))
ax.set_xticklabels(alphab)

plt.bar(pos, frequencies, width, color='r')
plt.show()
