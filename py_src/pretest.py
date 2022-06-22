import imp

import matplotlib.pyplot as plt

t = (0, 5, 15, 20)
s = (1.0, 0.4, 0.4, 1.0)

plt.plot(t, s)
plt.ylim(0.0, 1.2)
plt.xlabel("annealing_time")
plt.ylabel("annealing_schedule")
plt.savefig("as.png")
