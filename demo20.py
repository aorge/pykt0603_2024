
from matplotlib import pyplot as plt
import numpy as np

w0 = 0.25
w1 = 0.5
w2 = 1.0
w3 = 2.0
w4 = 3.0
message = 'w={}'

x = np.arange(-40, 40, 0.1)

for w in [w0, w1, w2, w3, w4]:
    f = 1 / (1 + np.exp(-x * w))
    plt.plot(x, f, label=message.format(w))
plt.legend(loc=2)
plt.show()