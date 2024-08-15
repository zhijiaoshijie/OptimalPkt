import matplotlib.pyplot as plt
x = [1, 2, 3]
y1 = [1, 2, 3]
y2 = [1, 2, 3]

fig1, ax1 = plt.subplots()
ax1.plot(x, y1)
ax1.set_title('Sine Wave')
ax1.set_xlabel('x')
ax1.set_ylabel('sin(x)')

fig2, ax2 = plt.subplots()
ax2.plot(x, y2)
ax2.set_title('Cosine Wave')
ax2.set_xlabel('x')
ax2.set_ylabel('cos(x)')

plt.show()
