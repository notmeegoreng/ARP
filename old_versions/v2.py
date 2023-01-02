import numpy as np
import sympy


"""
Similar to the ideas presented in v1
Implement a way to convert dice to the coefficients of the polynomial in e^t of the moment generating function 
This time, the coefficient arrays' first element does correspond to the chance of dealing 0 damage
Abandon when we realised with a chance of dealing zero damage, it meant that games could go on forever 
and so our current approach does not work
"""


def get_combined_die(dice: dict[int, int], hit_chance: sympy.Rational = sympy.Rational(1, 1)):
    """
    :param dice: dictionary of 'sides', 'number' pairs for kinds of dice
    :param hit_chance: chance of hitting, should be 0 < hit_chance <= 1
    :return: list where index represents relative probability of getting that much damage
    """
    # use moment generating functions
    # M_X(t)
    # we treat x = e^t
    x = sympy.symbols('x')
    total = 1
    for sides, n in dice.items():
        # mul the mgf of die type
        total *= sum(x ** i for i in range(1, sides + 1)) ** n
    # get coeffs starting with low powers first
    coefficients = sympy.Poly(total).all_coeffs()[::-1]
    coefficients[0] = sum(coefficients) * ((1 - hit_chance) / hit_chance)
    return coefficients


def damage_chances(coefficients, s, k):
    x = sympy.symbols('x')
    expr = sympy.Rational(1, s) * sum(c * x ** i for i, c in enumerate(coefficients))
    return sympy.Poly(sympy.expand_multinomial(expr ** k), x).all_coeffs()


def death_chance(coefficients, k, n):
    f = len(coefficients) - 1
    if n < k:
        return 0
    if n > k * f:
        return 1
    s = sum(coefficients)

    # negative index to get damages of >= n - f
    chances_damage = damage_chances(coefficients, s, k - 1)
    print(chances_damage, -n, f-n, chances_damage[-n:f - n])
    chances_damage = chances_damage[-n:f - n]
    chances_final_blow = sympy.Rational(1, s) * np.cumsum(coefficients[:0:-1])[::-1]
    print(chances_damage)
    return np.pad(chances_damage, (f - len(chances_damage), 0)) @ chances_final_blow


def death_chances(coefficients, n):
    """
    :param coefficients: list[int]: index of coefficients is damage dealt and value is relative probability
    :param n: int: amount of enemy health
    :return: chances, min_turns
    chances: list of chances of dying from turn number (min_turns) to turn number (n)
    min_turns: minimum turns to kill the enemy
    """
    s = sum(coefficients)  # coeff sum
    f = len(coefficients) - 1  # idr why i named this f, but its max damage per turn

    # mgf with x = e^t
    x = sympy.symbols('x')
    expr = sympy.Rational(1, s) * sum(c * x ** i for i, c in enumerate(coefficients))

    # get chance of dealing 'index' + 1 damage or more
    # remove chance of dealing 0 damage
    chances_final_blow = sympy.Rational(1, s) * np.cumsum(coefficients[:0:-1])[::-1]
    print(chances_final_blow)

    # minimum turns before a kill
    min_turns = (n - 1) // f + 1
    poly = sympy.Poly(expr ** (min_turns - 2))
    chances = []

    for k in range(min_turns, n + 1):
        poly *= expr
        # get chance of dealing enough damage to have health that gets destroyed in one hit
        chances_damage = poly.all_coeffs()[-n:f - n]
        # dot product with final blow chance to find chances of dying next turn
        chances.append(np.pad(chances_damage, (f - len(chances_damage), 0)) @ chances_final_blow)
    return chances, min_turns


def exp_and_var(coefficients, n):
    chances, k_initial = death_chances(coefficients, n)
    print(chances, k_initial)
    exp = exp2 = 0
    for k, c in enumerate(chances, k_initial):
        exp += k * c
        exp2 += k * k * c
    return exp, exp2 - exp * exp


# print(exp_and_var([1, 1], 6))
# print(death_chance([1, 1], 6, 7))
print(a := get_combined_die({4: 1, 6: 2, 10: 1}, sympy.Rational(1, 2)))
print(sum(a))
