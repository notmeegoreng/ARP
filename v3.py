import dataclasses

import numpy as np
import sympy


"""
Start using a Player class to hold data
Initial work on iteratively simulating turns one by one to deal with the possibly infinitely long games
Arbitrary precision could be reached by simply simulating more turns
"""


x = sympy.symbols('x')


def get_combined_die(dice: dict[int, int], hit_chance: sympy.Rational = sympy.Rational(1, 1)):
    """
    :param dice: dictionary of 'sides', 'number' pairs for kinds of dice
    :param hit_chance: chance of hitting, should be 0 < hit_chance <= 1
    :return: list where index represents relative probability of getting that much damage
    """
    # use moment generating functions
    # M_X(t)
    # we treat x = e^t
    total = 1
    for sides, n in dice.items():
        # mul the mgf of die type
        total *= sum(x ** i for i in range(1, sides + 1)) ** n
    # get coeffs starting with low powers first
    coefficients = sympy.Poly(total).all_coeffs()[::-1]
    coefficients[0] = sum(coefficients) * ((1 - hit_chance) / hit_chance)
    return coefficients


@dataclasses.dataclass
class Player:
    hp: int
    coefficients: np.ndarray[int]
    initiative_mod: int

    def get_expr(self):
        return sympy.Rational(1, np.sum(self.coefficients)) * sum(
            c * x ** i for i, c in enumerate(self.coefficients))

    def get_final_blow_chances(self):
        return sympy.Rational(1, np.sum(self.coefficients)) * np.cumsum(self.coefficients[:0:-1])[::-1]


def roll(a: Player, b: Player):
    a_max_dmg = len(a.coefficients) - 1
    b_max_dmg = len(b.coefficients) - 1
    # min turns until a player can deal a final blow
    min_turns = min((b.hp - 1) // a_max_dmg, (a.hp - 1) // b_max_dmg)

    # simulate up to min_turns
    a_expr = a.get_expr() ** min_turns
    b_expr = b.get_expr() ** min_turns

    # initiative
    diff = a.initiative_mod - b.initiative_mod
    if diff >= 20:
        ipa = 1
        ipb = 0
    elif diff <= -20:
        ipa = 0
        ipb = 1
    else:
        ipa = sympy.Rational((20 - diff) * (19 - diff), 800)
        ipb = 1 - ipa - sympy.Rational(20 - diff, 400)

    p = 0
    if ipa:
        # probability of a winning with a going first
        pass
    if ipb:
        # probability of a winning with b going first
        pass
    print(ipa, ipb)
    return p


roll(Player(8, np.array([1, 1]), -2), Player(4, np.array([1, 1, 1]), -5))
