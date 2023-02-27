from __future__ import annotations

import numpy as np


__all__ = ('linear_approximation', 'linear_predict')


def linear_approximation(damage: list | np.ndarray) -> tuple[float, float]:
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
    m = s / expected_damage_mul_s
    return m, np.sum(coeffs * np.arange(len(damage))) / expected_damage_mul_s * m


def linear_predict(damage: list | np.ndarray, hp: int):
    """Finds the approximated expected turns to defeat enemy"""
    m, c = linear_approximation(damage)
    return m * hp + c


if __name__ == '__main__':
    print(linear_approximation([0, 1, 2, 1]))
    print(linear_approximation([1, 1, 1]))
