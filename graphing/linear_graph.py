import numpy as np
import matplotlib.pyplot as plt

from linear_approximation import *
from recurrence import *

max_hp = 20
damage = [0, 1, 2, 1]

m, c = linear_approximation(damage)
print(m, c)
data_l = m * np.arange(max_hp + 1)
data_r = recurrence_method(damage, max_hp)
plt.plot(data_l, label='linear without extra term')
plt.plot(data_l + c, label='linear')
plt.plot(data_r, label='recurrence')

plt.title('Graph of E(T) against HP')
plt.legend()
plt.xlabel('HP')
plt.ylabel('E(T)')
plt.savefig('lin_graph.png')

print('done')
