# graph of variances
import time

import numpy as np
from matplotlib import pyplot as plt

import main

max_hp = 501

data = np.fromiter((main.expected_var(main.Player.from_coeffs(1, [1, 1], 1, 0), h)[1]
                    for h in range(5, max_hp)), float, max_hp - 5)

plt.plot(data)
plt.xlabel('HP')
plt.ylabel('Var(X)')
plt.savefig('graph_var.png')
plt.clf()
plt.plot(np.sqrt(np.arange(5, max_hp)), data)
plt.xlabel('sqrt HP')
plt.ylabel('E(X)')
plt.savefig('graph_sqrt_var.png')
print('done')


