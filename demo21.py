from matplotlib import pyplot as plt
import numpy as np

w1 = 3.0
bs = [-8, -4, 0, 4, 8]
message = "b={}"

x = np.arange(-10, 10, 0.1)
for b in bs:
    f = 1 / (1 + np.exp(-x * w1 + b))
    plt.plot(x, f, label=message.format(b))
plt.legend(loc=2)
plt.axvline(0)
plt.axhline(0.5)
plt.show()