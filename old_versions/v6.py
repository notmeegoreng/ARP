from __future__ import annotations

import dataclasses
import functools
from collections.abc import Sequence, Iterator

import numpy
import numpy as np
import sympy

x = sympy.symbols('x')


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
    # hits should not be <= 0
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


def binomial_distribution(n: int, p: sympy.Rational, s: int = 0, e: int | None = None) -> Iterator[sympy.Rational]:
    """Generates binomial distribution with n trials with successes ranging from s (default 0) to e (default n)"""
    if e is None:
        e = n
    v = (1 - p) ** (n - s) * p ** s
    if s != 0:
        for i in range(s + 1, n + 1):
            v *= i
            v /= i - s
    mod = p / (1 - p)
    yield v
    for i in range(s, min(n, e)):
        v *= mod
        v *= n - i
        v /= i + 1
        yield v
    for _ in range(n, e):
        yield 0


def _binomial_distribution(n: int, p: sympy.Rational, s: int = 0, e: int | None = None) -> Iterator[sympy.Rational]:
    """Generates binomial distribution with n trials with successes ranging from s (default 0) to e (default n)"""
    if e is None:
        e = n
    v = (1 - p) ** (n - s) * p ** s
    if s != 0:
        for i in range(s + 1, n + 1):
            v *= i
            v /= i - s
    mod = p / (1 - p)
    yield v
    for i in range(s, min(n, e)):
        v *= mod
        v *= n - i
        v /= i + 1
        yield v
    for _ in range(n, e):
        yield 0


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
    for binom in _binomial_distribution(turns, p.hit_chance):
        w_b = win_by(p.expr, hp, t)
        if w_b == 1:
            break
        left_binom -= binom
        total_win += w_b * binom
        t += 1
    return total_win + left_binom


def win_arrays(a: Player, b: Player, turns: int | None = None,
               exact: bool = True) -> tuple[sympy.Rational, sympy.Rational, sympy.Rational] | \
                                      tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]:
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
        raise ValueError('turns must be provided when hit chances are not 1!')
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
    return a_wins * (1 - b_wins_by), b_wins * (1 - a_wins_by), a_wins * b_wins


def win_chances(a: Player, b: Player, turns: int | None = None,
                exact: bool = True) -> tuple[sympy.Rational, sympy.Rational, sympy.Rational]:
    """
    Given two players, returns chances of the first player winning, the second player winning,
    or a draw (simultaneous elimination or no elimination).
    """
    a, b, d = win_arrays(a, b, turns, exact)
    return np.sum(a), np.sum(b), np.sum(d)  # type: ignore


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


def fix_length(dmg, length):
    if len(dmg) < length:
        return np.pad(dmg, (0, length - len(dmg)))
    elif len(dmg) > length:
        excess = np.sum(dmg[length:])
        dmg = dmg[:length]
        dmg[-1] += excess
        return dmg
    return dmg


def battle_outcomes(p_a: Player, p_b: Player, turns: int | None = None):
    """
    Takes in 2 player objects, and returns 2 arrays representing the chances
    of taking damage equal to the index for each player respectively.
    The last index represents the chance of losing: taking damage at least equal to their HP.
    """
    a_min_h, b_min_h = (p_b.hp - 1) // p_a.expr.degree(x) + 1, (p_a.hp - 1) // p_b.expr.degree(x) + 1
    a_max_h, b_max_h = p_b.hp + 1, p_a.hp + 1
    min_h = min(a_min_h, b_min_h)
    max_h = max(a_max_h, b_max_h)
    a_damage_rows_h = np.row_stack([
        fix_length(np.flip(damage_probabilities(p_a.expr, i).all_coeffs()), p_b.hp + 1)
        for i in range(min_h, max_h)
    ])
    b_damage_rows_h = np.row_stack([
        fix_length(np.flip(damage_probabilities(p_b.expr, i).all_coeffs()), p_a.hp + 1)
        for i in range(min_h, max_h)
    ])
    if turns is None and (p_a.hit_chance != p_b.hit_chance != 1):
        raise ValueError('turns must be provided when hit chances are not 1!')
    if turns < max_h:
        raise ValueError('turns too low, failing for accuracy and code simplification')
    print(min_h, max_h)
    if p_a.hit_chance == 1:
        # hits = turns
        if p_b.hit_chance == 1:
            a_damage_rows = a_damage_rows_h
        else:
            a_damage_rows = np.row_stack((
                a_damage_rows_h,
                np.zeros((turns - min_h - a_damage_rows_h.shape[0], p_b.hp + 1), dtype=object)
            ))
    else:
        a_damage_rows = np.empty((turns - min_h, p_b.hp + 1), dtype=sympy.Rational)
        for n in range(min_h, turns):
            a_hit_chances = np.fromiter(
                binomial_distribution(n, p_a.hit_chance, min_h, max_h), sympy.Rational, count=max_h - min_h)
            # print(n)
            # print(a_hit_chances)
            a_damage_rows[n - min_h] = np.sum(a_hit_chances[:, np.newaxis] * a_damage_rows_h, axis=0)
    print('A')
    print(a_damage_rows.shape)
    print(a_damage_rows)

    if p_b.hit_chance == 1:
        if p_a.hit_chance == 1:
            b_damage_rows = b_damage_rows_h
        else:
            b_damage_rows = np.row_stack((
                b_damage_rows_h,
                np.zeros((turns - min_h - b_damage_rows_h.shape[0], p_a.hp + 1), dtype=object)
            ))
    else:
        b_damage_rows = np.empty((turns - min_h, p_a.hp + 1), dtype=sympy.Rational)
        for n in range(min_h, turns):
            print(n)
            b_hit_chances = np.fromiter(
                binomial_distribution(n, p_b.hit_chance, min_h, max_h), sympy.Rational, count=max_h - min_h)
            print(b_hit_chances)
            b_damage_rows[n - min_h] = np.sum(b_hit_chances[:, np.newaxis] * b_damage_rows_h, axis=0)
    print('B')
    print(b_damage_rows.shape)
    print(b_damage_rows)

    a_wins = np.ediff1d(a_damage_rows[:, -1], to_begin=a_damage_rows[0, -1])
    b_wins = np.ediff1d(b_damage_rows[:, -1], to_begin=b_damage_rows[0, -1])
    print('wins')
    print(a_wins)
    print(b_wins)
    print('done')
    return (
        np.sum(a_wins[:, np.newaxis] * b_damage_rows, axis=0),
        np.sum(b_wins[:, np.newaxis] * a_damage_rows, axis=0),
    )


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


if __name__ == '__main__':
    '''
    a, b = battle_outcomes(
        Player.from_coeffs(10, get_combined_dice({2: 1}), sympy.Rational(1, 1), 0),
        Player.from_coeffs(20, get_combined_dice({2: 1}), sympy.Rational(1, 2), 0),
        100
    )
    print(a.astype(float))
    print(b.astype(float))
    '''

    print(expected_var(Player.from_coeffs(10, get_combined_dice({2: 2}), sympy.Rational(1, 2), 0), 25, 100))
