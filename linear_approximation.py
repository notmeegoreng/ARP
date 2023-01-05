import numpy as np


def r_term(damage: list):
    """
    :param damage: a list of relative chances of dealing damage equal to the index (one-indexed)
    :return: R term for the given damage chances.
    """
    s = np.sum(damage)
    coeffs = np.roll(s - np.cumsum(damage), 1)
    coeffs[0] = s
    expected_damage_mul_s = np.sum(np.arange(1, len(damage) + 1) * damage)
    return np.sum(coeffs * np.arange(len(damage))) / expected_damage_mul_s


def linear_approximation(damage):
    """
    :param damage: a list of relative chances of dealing damage equal to the index (one-indexed)
    :return: (m, c) tuple for use in the equation y = mx + c, where y is E(X) and x is HP.
    """
    return NotImplemented



if __name__ == '__main__':
    print(r_term([0, 1, 2, 1]))
