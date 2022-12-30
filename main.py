from __future__ import annotations

import dataclasses
import decimal
import functools
import time

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
    hit_chance: decimal.Decimal
    initiative_mod: int

    @classmethod
    def from_coeffs(cls, hp, coeffs, hit_chance, initiative_mod):
        return cls(hp, to_expr(coeffs), hit_chance, initiative_mod)


def get_combined_dice(dice: dict[int, int]) -> np.ndarray[int]:
    """
    :param dice: dictionary of 'sides', 'number' pairs for kinds of dice
    :return: list where one-indexed index represents relative probability of getting that much damage
    """
    # use moment generating functions
    # M_X(t)
    # we treat x = e^t
    total = 1
    for sides, n in dice.items():
        # mul the mgf of die type
        total *= sum(x ** i for i in range(1, sides + 1)) ** n
    # get coeffs starting with low powers first
    coefficients = sympy.Poly(total).all_coeffs()[:0:-1]
    return coefficients


def to_expr(coefficients):
    s = sum(coefficients)
    return sympy.Poly(sympy.Rational(1, s) * sum(c * x ** i for i, c in enumerate(coefficients, 1)))


cache = {}


def damage_probabilities(expr, turn):
    if expr not in cache:
        cache[expr] = [expr]
    for _ in range(len(cache[expr]), turn):
        cache[expr].append(cache[expr][-1] * expr)
    return cache[expr][turn - 1]


@functools.cache
def win_by(expr, n, t):
    if t >= n:
        return 1
    if t == 0:
        return 0
    return sum(damage_probabilities(expr, t).all_coeffs()[:-n])


def win_on(expr, n, t):
    return win_by(expr, n, t) - win_by(expr, n, t - 1)


def binomial_distribution(n, p):
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
def win_by_can_miss(p: Player, hp: int, turns: int):
    if p.hit_chance == 1:
        return win_by(p.expr, hp, turns)
    elif hp >= turns:
        return sum(win_by(p.expr, hp, t) * binom for t, binom in
                   zip(range(turns + 1), binomial_distribution(turns, p.hit_chance)))
    left_binom = 1
    total_win = 0
    for t, binom in zip(range(hp + 1), binomial_distribution(turns, p.hit_chance)):
        left_binom -= binom
        total_win += win_by(p.expr, hp, t) * binom
    print(total_win, left_binom, total_win + left_binom)
    return total_win + left_binom


def win_chances(a: Player, b: Player, turns: 'int | None' = None):
    """
    Given two players, returns chances of the first player winning, the second player winning,
    or a draw (simultaneous elimination or no elimination).
    """
    ex = True
    min_turns = min((b.hp - 1) // a.expr.degree(x), (a.hp - 1) // b.expr.degree(x)) + 1
    if turns < min_turns:
        return 0, 0, 1
    if a.hit_chance == b.hit_chance == 1:
        # case: all attacks hit
        # we can get exact numbers just by simulating up to min(a.hp, b.hp) + 1 turns,
        # when the fight is guaranteed to be over
        max_turns = min(a.hp, b.hp) + 1
        a_wins_by = np.fromiter((win_by(a.expr, b.hp, t) for t in range(min_turns, max_turns)),
                                sympy.Rational if ex else float, max_turns - min_turns)
        b_wins_by = np.fromiter((win_by(b.expr, a.hp, t) for t in range(min_turns, max_turns)),
                                sympy.Rational if ex else float, max_turns - min_turns)
    elif turns is None:
        raise ValueError('turns must be provided when hit chances are not 1')
    else:
        a_wins_by = np.fromiter((win_by_can_miss(a, b.hp, t) for t in range(min_turns, turns)),
                                sympy.Rational if ex else float, turns - min_turns)
        b_wins_by = np.fromiter((win_by_can_miss(b, a.hp, t) for t in range(min_turns, turns)),
                                sympy.Rational if ex else float, turns - min_turns)
    a_wins = np.ediff1d(a_wins_by, to_begin=a_wins_by[0])
    b_wins = np.ediff1d(b_wins_by, to_begin=b_wins_by[0])
    return (np.sum(a_wins * (1 - b_wins_by)),
            np.sum(b_wins * (1 - a_wins_by)),
            np.sum(a_wins * b_wins))


def win_chances_initiative(a: Player, b: Player, turns: int):
    # initiative calc
    diff = a.initiative_mod - b.initiative_mod
    # ip: initiative probability
    # how it works: each roll a d20, then add their initiative_mod
    # if equal, roll again
    # this can be modelled as chances of the difference between d20s being less than the difference in modifiers
    # difference between 2 independent d20s is a pyramid with linear sides, peak at 0
    if diff > 20:
        ipa = 1
        ipb = 0
    elif diff <= -20:
        ipa = 0
        ipb = 1
    elif diff > 0:
        # calc chance of Player a winning                      # normalise to 1, since ignoring equal case
        ipb = sympy.Rational((20 - diff) * (19 - diff), 800) / sympy.Rational(380 + diff, 400)
        ipa = 1 - ipb
    else:
        ipa = sympy.Rational((20 + diff) * (19 + diff), 800) / sympy.Rational(380 - diff, 400)
        ipb = 1 - ipa

    a_w, b_w, d = win_chances(a, b, turns)
    return a_w + ipa * d, b_w + ipb * d


def expected_var(p: Player, hp: int, turns: int | None = None):
    min_turns = (hp - 1) // p.expr.degree(x) + 1
    ex = True
    if p.hit_chance == 1:
        turns = hp + 1
        wins_by = np.fromiter((win_by(p.expr, hp, t) for t in range(min_turns, turns)),
                              sympy.Rational if ex else float, turns - min_turns)
    elif turns is None:
        raise ValueError('turns must be provided when hit chances are not 1')
    else:
        wins_by = np.fromiter((win_by_can_miss(p, hp, t) for t in range(min_turns, turns)),
                              sympy.Rational if ex else float, turns - min_turns)

    wins = np.ediff1d(wins_by, to_begin=wins_by[0])
    exp = np.arange(min_turns, turns) * wins
    return np.sum(exp), np.sum(exp * wins)


def main():
    player_dice = ({2: 1}, {4: 1}, {6: 1}, {4: 2}, {6: 2}, {10: 1}, {6: 1, 1: 3})
    players = (Player.from_coeffs(1, get_combined_dice(dice), 1, 1) for dice in player_dice)

    start = time.perf_counter_ns()
    print('Players\n')
    for i, dice in enumerate(player_dice, 1):
        print(f'Player {i} dice: {dice}')
    for i, p in enumerate(players, 1):
        print(f'\nPlayer {i}')
        for hp in (10, 20, 30, 40, 50, 60, 100):
            exp, var = expected_var(p, hp)
            print(f'{hp} HP: E: {float(exp)}, V: {float(var)}')
    end = time.perf_counter_ns()
    print(end - start)

    start = time.perf_counter_ns()

    end = time.perf_counter_ns()
    print(end - start)

    """
    p0 = Player.from_coeffs(12, (1, 1), sympy.Rational(1, 2), 50)
    p1 = Player.from_coeffs(13, (1,), sympy.Rational(1, 2), 50)

    # r = win_chances_initiative(p0, p1, 20)
    print(win_by(p0.expr, 10, 7))
    start = time.perf_counter_ns()
    r = win_chances(p0, p1, 100)
    end = time.perf_counter_ns()
    print(end - start)
    print(r)
    print(*(float(a) for a in r))
    print(err := 1 - sum(r))  # error
    print(float(err))
    """


if __name__ == '__main__':
    main()
