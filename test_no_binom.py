import time

import sympy

from main_ext import *

poly = get_combined_dice({2: 2})
poly[0] = sympy.Rational(1, 1)
poly /= 2
print(poly)
out = damage_probabilities(poly, 2)
print(out)
start = time.perf_counter_ns()
print(d := damage_probabilities(poly, 12))
end = time.perf_counter_ns()
print(end - start)
print(d.shape)

start = time.perf_counter_ns()
out = win_by_attacks({Attack(get_combined_dice({2: 2}), sympy.Rational(1, 2), 20): 12}, 50)
end = time.perf_counter_ns()
print(end - start)
print(out)


import main


start = time.perf_counter_ns()
out = main.win_chances(
    main.Player.from_coeffs(3, get_combined_dice({3: 1}), sympy.Rational(1, 2), 1),
    main.Player.from_coeffs(3, get_combined_dice({3: 1}), sympy.Rational(1, 2), 1),
    3
)
end = time.perf_counter_ns()
print(end - start)
print(out)
print([float(x) for x in out])
