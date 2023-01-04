import numpy as np
import sympy


def recurrence_method(damage, turn):
    """Uses the recurrence relationship to calculate E(X)"""
    # recurrence relationship -
    length = len(damage)
    s = sum(damage)
    # index in data is turn offset by -length
    data = np.empty(turn + length + 1, dtype=float)
    data[:length + 1] = 0
    for i in range(length + 1, turn + length + 1):
        data[i] = np.sum(data[i - length:i] * damage) / s + 1

    print(data)
    return data[-1]


print(recurrence_method(np.array([1, 1], dtype=int), 7))
