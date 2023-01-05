import numpy as np
import sympy


def recurrence_method(damage: np.ndarray, turn):
    """
    :param damage: a list of relative chances of dealing damage equal to the index (one-indexed)
    :param turn: an integer number of turns
    :return: array of E(X), up to turn, calculated with the recurrence relationship.
    """
    # recurrence relationship -
    length = len(damage)
    s = sum(damage)
    # index in data is turn offset by -length
    data = np.empty(turn + length + 1, dtype=float)
    data[:length + 1] = 0
    for i in range(length + 1, turn + length + 1):
        data[i] = np.sum(data[i - length:i] * damage) / s + 1

    return data


if __name__ == '__main__':
    print(recurrence_method(np.array([1, 1], dtype=int), 7))
