import imp

import matplotlib.pyplot as plt
from numpy import arange, log, tanh


def f(x):
    t = 1000000000
    p = 8
    return - t / 2. * log(1.0 / tanh(x / p * t))


x = arange(1e-10, 3, 0.00001)
y = f(x)

plt.plot(x, y)
plt.show()
