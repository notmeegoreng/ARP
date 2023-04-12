from __future__ import annotations

import dataclasses
import functools
import time
from collections.abc import Sequence, Iterator

import numpy as np
import sympy

x = sympy.symbols('x')


"""
Optimise to not need to calc the rest of the binomial series when the number of hits 
ensures that chance to kill is 1
"""


@dataclasses.dataclass(frozen=True)
class Player:
    hp: int
    expr: sympy.Poly
    hit_chance: sympy.Rational
    initiative_mod: int

    @classmethod
    def from_coeffs(cls, hp, coeffs, hit_chance, initiative_mod) -> 'Player':
        return cls(hp, to_expr(coeffs), hit_chance, initiative_mod)


def get_combined_dice(dice: dict[int, int]) -> list[int]:
    """
    :param dice: dictionary of 'sides', 'number' pairs for kinds of dice
    :return: list where one-indexed index represents relative probability of getting that much damage
    """
    total = 1  # total pgf
    for sides, n in dice.items():
        # mul the pgf of this dice type
        total *= sum(x ** i for i in range(1, sides + 1)) ** n

    # sympy returns coeffs with the highest power first
    # get coeffs starting with low powers first, and remove the coeff of x^0 (which will be 0)
    coefficients = sympy.Poly(total).all_coeffs()[-2::-1]
    return coefficients


def to_expr(coefficients: Sequence[int]) -> sympy.Poly:
    s = sum(coefficients)
    return sympy.Poly(sympy.Rational(1, s) * sum(c * x ** i for i, c in enumerate(coefficients, 1)))


cache = {}


def damage_probabilities(expr: sympy.Poly, hits: int) -> sympy.Poly:
    # turn should not be <= 0
    if expr not in cache:
        cache[expr] = [expr]
    for _ in range(len(cache[expr]), hits):
        cache[expr].append(cache[expr][-1] * expr)
    return cache[expr][hits - 1]


@functools.cache
def win_by(expr: sympy.Poly, n: int, hits: int) -> sympy.Rational | int:
    if hits >= n:
        return 1
    if hits == 0:
        return 0
    return sum(damage_probabilities(expr, hits).all_coeffs()[:-n])


def win_on(expr: sympy.Poly, n: int, t: int) -> sympy.Rational | int:
    return win_by(expr, n, t) - win_by(expr, n, t - 1)


def binomial_distribution(n: int, p: sympy.Rational) -> Iterator[sympy.Rational]:
    """Generates binomial distribution for N=0 to N=N"""
    v = (1 - p) ** n
    mod = p / (1 - p)
    yield v
    for i in range(0, n):
        v *= mod
        v *= n - i
        v /= i + 1
        yield v


@functools.cache
def win_by_can_miss(p: Player, hp: int, turns: int) -> sympy.Rational | int:
    if p.hit_chance == 1:
        return win_by(p.expr, hp, turns)

    # sum the binomial probability of hitting some number of times * chance of winning by that number of hits
    # once enough hits that its enough to confirm win, the second term above will always be 1 for rest of terms
    # just sum the rest of the binomial probability
    # (taking advantage of fact that binomial probability sums to 1)
    left_binom = 1
    total_win = 0
    t = 0
    for binom in binomial_distribution(turns, p.hit_chance):
        w_b = win_by(p.expr, hp, t)
        if w_b == 1:
            break
        left_binom -= binom
        total_win += w_b * binom
        t += 1
    return total_win + left_binom


def win_chances(a: Player, b: Player, turns: int | None = None, exact: bool = True
                ) -> tuple[sympy.Rational, sympy.Rational, sympy.Rational]:
    """
    Given two players, returns chances of the first player winning, the second player winning,
    or a draw (simultaneous elimination or no elimination).
    """

    min_turns = min((b.hp - 1) // a.expr.degree(x), (a.hp - 1) // b.expr.degree(x)) + 1
    if turns < min_turns:
        return sympy.Rational(0, 1), sympy.Rational(0, 1), sympy.Rational(1, 1)
    if a.hit_chance == b.hit_chance == 1:
        # case: all attacks hit
        # we can get exact numbers just by simulating up to min(a.hp, b.hp) + 1 turns,
        # when the fight is guaranteed to be over
        max_turns = min(a.hp, b.hp) + 1
        a_wins_by = np.fromiter((win_by(a.expr, b.hp, t) for t in range(min_turns, max_turns)),
                                sympy.Rational if exact else float, max_turns - min_turns)
        b_wins_by = np.fromiter((win_by(b.expr, a.hp, t) for t in range(min_turns, max_turns)),
                                sympy.Rational if exact else float, max_turns - min_turns)
    elif turns is None:
        raise ValueError('turns must be provided when hit chances are not 1')
    else:
        # when attacks have a hit chance, the battle could theoretically go on forever
        # we calculate up to a given number of turns
        a_wins_by = np.fromiter((win_by_can_miss(a, b.hp, t) for t in range(min_turns, turns)),
                                sympy.Rational if exact else float, turns - min_turns)
        b_wins_by = np.fromiter((win_by_can_miss(b, a.hp, t) for t in range(min_turns, turns)),
                                sympy.Rational if exact else float, turns - min_turns)
    # chance of winning on turn n = chance of winning by turn n - chance of winning by turn n-1
    a_wins = np.ediff1d(a_wins_by, to_begin=a_wins_by[0])
    b_wins = np.ediff1d(b_wins_by, to_begin=b_wins_by[0])

    # chance of a player winning each turn:
    # chance they defeat other on that turn * chance that they have not been defeated
    # chance of simultaneous elimination each turn: chance each defeats other multiplied together
    return (np.sum(a_wins * (1 - b_wins_by)),  # type: ignore
            np.sum(b_wins * (1 - a_wins_by)),
            np.sum(a_wins * b_wins))


def win_chances_initiative(a: Player, b: Player, turns: int) -> tuple[sympy.Rational, sympy.Rational]:
    # initiative calc
    diff = a.initiative_mod - b.initiative_mod
    # ip: initiative probability
    # how it works: each roll a d20, then add their initiative_mod
    # if equal, roll again
    # this can be modelled as chances of the difference between d20s being less than the difference in modifiers
    # difference between 2 independent d20s is a pyramid with linear sides, peak at 0
    if diff >= 20:
        ipa = 1
        ipb = 0
    elif diff <= -20:
        ipa = 0
        ipb = 1
    elif diff > 0:
        # calc chance of Player a winning and normalise to 1, since ignoring equal case
        # calc: (20 - d)(19 - d)/800 * (380 + d)/400 where d is absolute difference
        ipb = sympy.Rational((20 - diff) * (19 - diff), 380 + diff) / 2
        ipa = 1 - ipb
    else:
        ipa = sympy.Rational((20 + diff) * (19 + diff), 380 - diff) / 2
        ipb = 1 - ipa

    # first to attack wins in case of simultaneous elimination
    a_w, b_w, d = win_chances(a, b, turns)
    return a_w + ipa * d, b_w + ipb * d


def expected_var(p: Player, hp: int, turns: int | None = None, exact: bool = True
                 ) -> tuple[sympy.Rational, sympy.Rational]:
    min_turns = (hp - 1) // p.expr.degree(x) + 1

    if p.hit_chance == 1:
        turns = hp + 1
        wins_by = np.fromiter((win_by(p.expr, hp, t) for t in range(min_turns, turns)),
                              sympy.Rational if exact else float, turns - min_turns)
    elif turns is None:
        raise ValueError('turns must be provided when hit chances are not 1')
    else:
        wins_by = np.fromiter((win_by_can_miss(p, hp, t) for t in range(min_turns, turns)),
                              sympy.Rational if exact else float, turns - min_turns)

    wins = np.ediff1d(wins_by, to_begin=wins_by[0])
    exp = np.arange(min_turns, turns) * wins
    return np.sum(exp), np.sum(exp * wins)  # type: ignore
