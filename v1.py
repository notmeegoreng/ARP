import numpy as np
import sympy


"""
Here we calculate chance of dying on a turn by calculating up to the turn before, 
then finding out the chance of dealing enough to finish the opponent off
from there we get the expected value and variance for approximation
"""


def damage_chances(coefficients, s, k):
    x = sympy.symbols('x')
    expr = sympy.Rational(1, s) * sum(c * x ** i for i, c in enumerate(coefficients, 1))
    # negative index to get damages of >= n - f
    return sympy.Poly(sympy.expand_multinomial(expr ** k), x).all_coeffs()


def death_chance(coefficients, k, n):
    f = len(coefficients)
    if n < k:
        return 0
    if n > k * f:
        return 1
    s = sum(coefficients)

    # negative index to get damages of >= n - f
    chances_damage = damage_chances(coefficients, s, k - 1)
    print(chances_damage, -n, f-n, chances_damage[-n:f - n])
    chances_damage = chances_damage[-n:f - n]
    chances_final_blow = sympy.Rational(1, s) * np.cumsum(coefficients[::-1])[::-1]
    print(chances_damage)
    return np.dot(np.pad(chances_damage, (f - len(chances_damage), 0)), chances_final_blow)


def death_chances(coefficients, n):
    s = sum(coefficients)
    f = len(coefficients)
    x = sympy.symbols('x')
    expr = sympy.Rational(1, s) * sum(c * x ** i for i, c in enumerate(coefficients, 1))
    chances_final_blow = sympy.Rational(1, s) * np.cumsum(coefficients[::-1])[::-1]
    print(chances_final_blow)
    min_turns = (n - 1) // f + 1
    poly = sympy.Poly(expr ** (min_turns - 2))
    chances = []
    for k in range(min_turns, n + 1):
        poly *= expr
        chances_damage = poly.all_coeffs()[-n:f - n]
        chances.append(np.dot(np.pad(chances_damage, (f - len(chances_damage), 0)), chances_final_blow))
    return chances, min_turns


def exp_and_var(coefficients, n):
    chances, k_initial = death_chances(coefficients, n)
    print(chances, k_initial)
    exp = exp2 = 0
    for k, c in enumerate(chances, k_initial):
        exp += k * c
        exp2 += k * k * c
    return exp, exp2 - exp * exp


print(exp_and_var([1, 1], 24))
