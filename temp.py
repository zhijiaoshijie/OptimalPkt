import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 10, 100)
y = np.sin(x)

fig, ax = plt.subplots()
ax.plot(x, y)

# Enable the toolbar for zooming and panning
plt.show()

import matplotlib
matplotlib.use('Qt5Agg')  # or 'TkAgg', 'WebAgg', etc.
