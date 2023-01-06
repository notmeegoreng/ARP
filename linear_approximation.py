import numpy as np


def linear_approximation(damage: list) -> tuple[float, float]:
    """
    :param damage: a list of relative chances of dealing damage equal to the index (one-indexed)
    :return: (m, c) tuple for use in the equation y = mx + c, where y is E(X) and x is HP.
    """
    s = np.sum(damage)
    coeffs = np.roll(s - np.cumsum(damage), 1)  # for R term calculation
    assert coeffs[0] == 0
    coeffs[0] = s
    # the expected damage each turn
    expected_damage_mul_s = np.sum(np.arange(1, len(damage) + 1) * damage)
    return expected_damage_mul_s / s, np.sum(coeffs * np.arange(len(damage))) / expected_damage_mul_s


if __name__ == '__main__':
    print(linear_approximation([0, 1, 2, 1]))
    print(linear_approximation([1, 1, 1]))
