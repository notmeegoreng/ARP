import numpy as np
from fractions import Fraction

import test


"""
This algorithm relies on variations of Pascal's Triangle 
where the initial row is the coeffs of the mgf of damage dealt
it was abandoned when we realised that our method of calculating coefficients from previous rows was not generalisable
"""


class Algo:
    def __init__(self, faces):
        self.f = faces
        self.cache = [np.zeros((), dtype=int), np.ones((faces,), dtype=int)]
        self.done = 1

    def find(self, k):
        if k <= self.done:
            return self.cache[k]
        self.populate(k)
        return self.cache[k]

    def populate(self, k):
        for i in range(self.done + 1, k + 1):
            prev = self.cache[-1]
            len_expected = len(prev) + self.f - 1
            row = np.empty((len_expected,), dtype=int)
            row[0] = prev[0]
            len_expected = len(prev) + self.f - 1
            half = (len_expected + 1) // 2
            for j in range(1, len_expected):
                row[j] = prev[j] + row[j - 1] - (prev[j - self.f] if len(prev) > j - self.f >= 0 else 0)
            # row[half:] = row[half - 1 - len_expected % 2::-1]
            self.cache.append(row)

    def chance_of_loss(self, n, k):
        if k >= n:
            return Fraction(1)
        if n > k * self.f:
            return Fraction(0)

        # damage values: k to self.f * (k - 1)
        p = self.find(k - 1)
        # type checker does not know np.dot returns scalar in this case
        return Fraction(np.dot(p[n - 2 * self.f:], np.arange(1, self.f + 1)), self.f ** k)  # type: ignore


if __name__ == '__main__':
    f = 4
    test.run([1] * f, 6)
    a = Algo(f)
    a.find(6)
    # print(a.chance_of_loss(15, 4))
    print('\n'.join(map(str, a.cache)))
