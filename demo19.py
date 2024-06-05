from matplotlib import pyplot as plt
import numpy as np

x = np.arange(-10, 10, 0.1)

f = 1 / (1 + np.exp(-x))

plt.xlabel('x')
plt.ylabel('F(X)')
plt.plot(x, f)
plt.axvline(0)
plt.axhline(0.5)
plt.show()