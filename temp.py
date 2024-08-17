import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-1, 1, 100)
y = x**2

fig, ax = plt.subplots()
ax.plot(x, y)
plt.show()
fig, ax = plt.subplots()
x = np.linspace(-1, 1, 100)
y = x**3
ax.plot(x, y)
plt.show()