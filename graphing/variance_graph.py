# graph of variances

import numpy as np
from matplotlib import pyplot as plt

import main

min_hp = 1
max_hp = 10

data = np.fromiter((main.expected_var(main.Player.from_coeffs(1, [1, 1], 0.5, 0), h, 300)[1]
                    for h in range(min_hp, max_hp + 1)), float, max_hp - min_hp + 1)

plt.plot(np.arange(min_hp, max_hp + 1), data)
plt.xlabel('HP')
plt.ylabel('Var(T)')
plt.title('Graph of Var(T) against HP ')
plt.savefig('graph_var.png')
plt.clf()
plt.plot(np.sqrt(np.arange(min_hp, max_hp + 1)), data)
plt.xlabel('sqrt HP')
plt.ylabel('Var(T)')
plt.title('Graph of Var(T) against sqrt(HP)')
plt.savefig('graph_sqrt_var.png')
print('done')


