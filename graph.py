import numpy as np
import matplotlib.pyplot as plt

data = np.fromfunction(lambda x: -2*(-1/2)**x/9 + 2*x/3 + 2/9, (51,))

plt.plot(data)
plt.xlabel('HP')
plt.ylabel('E(X)')
plt.savefig('graph0.png')
plt.clf()
plt.plot(data[:6])
plt.xlabel('HP')
plt.ylabel('E(X)')
plt.savefig('graph1.png')

