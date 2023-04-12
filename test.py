import time

import sympy

from main import Player, get_combined_dice, win_chances, win_chances_initiative, expected_var
from recurrence import recurrence_method
from linear_approximation import linear_predict

p0 = Player.from_coeffs(20, get_combined_dice({2: 2}), sympy.Rational(2, 3), 0)
p1 = Player.from_coeffs(15, get_combined_dice({2: 3}), sympy.Rational(1, 1), 0)

# ex0, var0 = expected_var(p0, p1.hp, 100)

start = time.perf_counter_ns()
print(linear_predict(get_combined_dice({2: 2}), p1.hp))
end = time.perf_counter_ns()
print('A time:', end - start)

start = time.perf_counter_ns()
print(recurrence_method(get_combined_dice({2: 2}), p1.hp)[-1])
end = time.perf_counter_ns()
print('B time:', end - start)

start = time.perf_counter_ns()
print(*map(float, win_chances_initiative(p0, p1, 1000)))
end = time.perf_counter_ns()


print('C time:', end - start)

#
# print(1 - sum(win_chances(p0, p1, 1000)))
