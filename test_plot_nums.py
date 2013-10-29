import numpy as np
import matplotlib.pyplot as plt

frequencies = [23, 44, 12, 11, 2, 10]

ax = plt.axes()
ax.set_xticks(pos + (width / 2))
ax.set_xticklabels(alphab)

plt.bar(pos, frequencies, width, color='r')
plt.show()



