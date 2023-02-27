import numpy as np

import main


__all__ = ('recurrence_method',)


def recurrence_method(damage: list, max_hp: int):
    """
    :param damage: a list of relative chances of dealing damage equal to the index (one-indexed)
    :param max_hp: an integer number of turns
    :return: array of E(X) for HP ranging from -(max_damage + 1) up to max_hp,
     calculated with the recurrence relationship.
    """
    # recurrence relationship -
    length = len(damage)
    s = sum(damage)
    # index in data is turn offset by -length
    data = np.empty(max_hp + length + 1, dtype=float)
    data[:length + 1] = 0
    for i in range(length + 1, max_hp + length + 1):
        data[i] = np.sum(data[i - 1:i - length - 1:-1] * damage) / s + 1

    return data[length:]


if __name__ == '__main__':
    print(a := recurrence_method([0, 1, 2, 1], 10))
    print(len(a))
    print(recurrence_method([1, 1], 7))
    arr = recurrence_method(main.get_combined_dice({1: 18, 6: 2, 8: 2, 10: 2}), 207)
    print(arr[-1])
    print(arr[-(207 - 172) - 1])
