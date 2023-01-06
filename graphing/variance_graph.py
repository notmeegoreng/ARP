# graph of variances

import numpy as np
from matplotlib import pyplot as plt

import main

max_hp = 101

data = np.fromiter((main.expected_var(main.Player.from_coeffs(1, [1, 1], 0.5, 0), h, 300)[1]
                    for h in range(5, max_hp)), float, max_hp - 5)

plt.plot(np.arange(5, max_hp), data)
plt.xlabel('HP')
plt.ylabel('Var(X)')
plt.savefig('graph_var.png')
plt.clf()
plt.plot(np.sqrt(np.arange(5, max_hp)), data)
plt.xlabel('sqrt HP')
plt.ylabel('Var(X)')
plt.savefig('graph_sqrt_var.png')
print('done')


