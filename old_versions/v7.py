from __future__ import annotations

import dataclasses
import functools
import itertools
from collections.abc import Sequence, Iterator, Iterable, Callable

import numpy
import numpy as np
import sympy

x = sympy.symbols('x')


@dataclasses.dataclass(frozen=True)
class Attack:
    poly: np.ndarray
    hit_chance: sympy.Rational
    uses: int

    def expected_dmg(self):
        return self.hit_chance * (self.poly * np.arange(len(self.poly))).sum()

    def __hash__(self):
        return hash((self.poly.tobytes(), self.hit_chance, self.uses))


@dataclasses.dataclass(frozen=True)
class Player:
    hp: int
    attacks: tuple[Attack, ...]
    initiative_mod: int

    def __post_init__(self):
        # sort attacks by expected damage
        object.__setattr__(
            self, 'attacks',
            tuple(sorted(self.attacks, key=lambda att: att.expected_dmg(), reverse=True)))


def get_combined_dice(dice: dict[int, int]) -> np.ndarray:
    """
    :param dice: dictionary of 'sides', 'number' pairs for kinds of dice
    :return: list where one-indexed index represents relative probability of getting that much damage
    """
    total = 1  # total pgf
    for sides, n in dice.items():
        # mul the pgf of this dice type
        total *= sum(x ** i for i in range(1, sides + 1)) ** n

    # sympy returns coeffs with the highest power first
    # get coeffs starting with low powers first
    coefficients = np.array(sympy.Poly(total).all_coeffs()[-1::-1], dtype=sympy.Rational)
    coefficients /= coefficients.sum()
    return coefficients


def to_expr(coefficients: Sequence[int]) -> sympy.Poly:
    s = sum(coefficients)
    return sympy.Poly(sympy.Rational(1, s) * sum(c * x ** i for i, c in enumerate(coefficients, 1)))


def poly_mul(poly0: np.ndarray, poly1: np.ndarray) -> np.ndarray:
    """
    Takes in two vectors representing polynomials and return a vector representing their multiplication.
    One of the vectors can be a matrix, which would be considered as an array of polynomials.
    A matrix representing an array of output polynomials is returned.

    converting polynomial multiplication to matrix
    credit https://www.le.ac.uk/users/dsgp1/COURSES/THIRDMET/MYLECTURES/9XMATRIPOL.pdf
    fast rolling to construct the matrix required
    credit https://stackoverflow.com/a/42101326
    """
    if len(poly0.shape) > 1:
        if len(poly1.shape) > 1:
            raise ValueError('Multiple multidimensional arguments not supported!')
        poly0, poly1 = poly1, poly0
    if len(poly0) == 1 or poly1.shape[-1] == 1:
        return poly0 * poly1
    le = poly1.shape[-1]
    mat = np.repeat(np.pad(poly0, (0, le - 1))[np.newaxis, :], le, axis=0)
    m, n = mat.shape
    idx = np.mod((n - 1) * np.arange(m)[:, np.newaxis] + np.arange(n), n)
    out = poly1 @ mat[np.arange(m)[:, np.newaxis], idx]
    return out


cache = {}


def damage_probabilities(base: np.ndarray, hits: int) -> np.ndarray:
    if hits < 0:
        raise ValueError('hits >= 0!')
    if hits == 0:
        return np.array((1,))
    h = hash(base.tobytes())
    if h not in cache:
        cache[h] = [base]
    for _ in range(len(cache[h]), hits):
        cache[h].append(poly_mul(cache[h][-1], base))
    return cache[h][hits - 1]


def win_by(poly: np.ndarray, n: int) -> sympy.Rational:
    if len(poly) == 1:
        return sympy.Rational(0, 1)
    return poly[n:].sum()


def win_on(poly: np.ndarray, n: int, t: int) -> sympy.Rational:
    return win_by(damage_probabilities(poly, t), n) - win_by(damage_probabilities(poly, t - 1), n)


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


def get_attacks(p: Player, turns: int) -> dict[Attack, int]:
    # greedy algo for now
    attacks = {}
    for att in p.attacks:
        if att.uses == 0:
            attacks[att] = turns
            return attacks
        elif turns >= att.uses:
            attacks[att] = att.uses
            if turns == att.uses:
                return attacks
            turns -= att.uses
        else:
            attacks[att] = turns
            return attacks
    raise ValueError('Not enough attacks to use for that many turns!')


def win_by_attacks(attacks: dict[Attack, int], hp: int) -> sympy.Rational:
    print(attacks)
    poly_base = np.ones(1, dtype=sympy.Rational)
    # per attack binomials
    attacks_b = []
    binomials: list[Iterator[sympy.Rational]] = []
    for attack, times in attacks.items():
        if attack.hit_chance == 1:
            poly_base = poly_mul(poly_base, damage_probabilities(attack.poly, times))
        else:
            attacks_b.append(attack)
            binomials.append(enumerate(binomial_distribution(times, attack.hit_chance)))
    print(attacks_b)
    print('base:',  )
    if binomials:
        total = 0
        for binomials in itertools.product(*binomials):
            poly = poly_base
            coeff = 1
            for att, (times, binom) in zip(attacks_b, binomials):
                poly = poly_mul(poly, damage_probabilities(att.poly, times))
                coeff *= binom
            print(poly)
            w_b = win_by(poly, hp)
            if w_b == 1:
                # possible optimisation here
                pass
            total += w_b * coeff
        print(total)
        return total  # type: ignore
    return win_by(poly_base, hp)


@functools.cache
def win_by_can_miss(p: Player, turns: int, hp: int) -> sympy.Rational:
    return win_by_attacks(get_attacks(p, turns), hp)


def get_min_turns(p: Player, hp: int):
    turns = 0
    for att in p.attacks:
        max_dmg = len(att.poly)
        if att.uses != 0 and max_dmg * att.uses <= hp:
            turns += att.uses
            hp -= max_dmg * att.uses
            if hp == 0:
                return turns
        else:
            return turns + (hp - 1) // max_dmg + 1


def win_arrays(p0: Player, p1: Player, turns: int,
               exact: bool = True) -> tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]:
    min_turns = min(get_min_turns(p0, p1.hp), get_min_turns(p1, p0.hp))
    if turns < min_turns:
        raise ValueError('Not enough turns - confirmed draw')

    # when attacks have a hit chance, the battle could theoretically go on forever
    # we calculate up to a given number of turns
    wins_by0 = np.fromiter((win_by_can_miss(p0, t, p1.hp) for t in range(min_turns, turns)),
                           sympy.Rational if exact else float, turns - min_turns)
    wins_by1 = np.fromiter((win_by_can_miss(p1, t, p0.hp) for t in range(min_turns, turns)),
                           sympy.Rational if exact else float, turns - min_turns)
    # chance of winning on turn n = chance of winning by turn n - chance of winning by turn n-1
    wins0 = np.ediff1d(wins_by0, to_begin=wins_by0[0])
    wins1 = np.ediff1d(wins_by1, to_begin=wins_by1[0])

    # chance of a player winning each turn:
    # chance they defeat other on that turn * chance that they have not been defeated
    # chance of simultaneous elimination each turn: chance each defeats other multiplied together
    return wins0 * (1 - wins_by1), wins1 * (1 - wins_by0), wins0 * wins1


if __name__ == '__main__' and False:
    p0, p1, d = win_arrays(
        Player(13, (Attack(np.array(get_combined_dice({2: 1})), sympy.Rational(1, 1), 0),), -1),
        Player(12, (Attack(np.array(get_combined_dice({2: 2, 1: 1})), sympy.Rational(1, 2), 2),
                    Attack(np.array(get_combined_dice({2: 1})), sympy.Rational(1, 1), 0)), 1),
        30
    )
    print(p0)
    print(p1)
    print(d)


def win_chances(p0: Player, p1: Player, turns: int,
                exact: bool = True) -> tuple[sympy.Rational, sympy.Rational, sympy.Rational]:
    """
    Given two players, returns chances of the first player winning, the second player winning,
    or a draw (simultaneous elimination or no elimination).
    """
    w0, w1, d = win_arrays(p0, p1, turns, exact)
    return np.sum(w0), np.sum(w1), np.sum(d)  # type: ignore


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


def fix_width(mat: np.ndarray, width: int) -> np.ndarray:
    curr = mat.shape[-1]
    if curr < width:
        return np.pad(mat, ((0, 0), (0, width - curr)))
    elif curr > width:
        excess = mat[:, width:].sum(axis=1)
        mat = mat[:, :width]
        mat[:, -1] += excess
        return mat
    return mat


def _pow_range_gen(p, e):
    p_i = p
    yield 1
    for _ in range(e):
        yield p
        p *= p_i


def _binomial_2d_helper(n, p) -> Iterable[np.ndarray]:
    pow_arr = np.fromiter(_pow_range_gen(1 - p, n), count=n + 1, dtype=sympy.Rational)
    yield pow_arr
    col1 = np.arange(n + 1, dtype=sympy.Rational)
    col1[1:] *= pow_arr[:-1]
    yield col1
    p_i = p
    for i in range(2, n + 1):
        col = np.zeros(n + 1, dtype=sympy.Rational)
        p_i *= p
        col[i] = p_i
        for j in range(i + 1, n + 1):
            #           prev    |   coeff
            col[j] = col[j - 1] * j / (j - i)
        col[i:] *= pow_arr[:-i]
        yield col


@functools.cache
def binomial_2d(n, p):
    return np.column_stack(tuple(_binomial_2d_helper(n, p)))


def attack_outcomes(attack: Attack, uses: int, limit: int):
    # omit first row of all zeros for eff
    hit_damage_rows = np.row_stack((
        np.zeros(limit, dtype=sympy.Rational),
        np.row_stack([
            fix_length(damage_probabilities(attack.poly, i), limit)
            for i in range(1, uses + 1)
        ])
    ))
    hit_chances = binomial_2d(uses, attack.hit_chance)
    print(hit_chances)
    print(hit_damage_rows)
    damage_rows = np.sum(hit_chances[:, :, np.newaxis] * hit_damage_rows, axis=1)
    return damage_rows


def battle_outcomes(hp0: int, attacks0: dict[Attack, int],
                    hp1: int, attacks1: dict[Attack, int],
                    limit0: int | None = None, limit1: int | None = None):
    """
    Takes in 2 player objects, and returns 2 arrays representing the chances
    of taking damage equal to the index for each player respectively.
    The last index represents the chance of losing: taking damage at least equal to their HP.
    """
    all_damage_rows = []
    wins = []
    for e_h, attacks, limit in ((hp1, attacks0, limit0), (hp0, attacks1, limit1)):
        e_h += 1
        base = np.ones(1, dtype=sympy.Rational)
        array_list = [np.zeros(e_h)]
        for attack, times in attacks.items():
            print(attack)
            if limit is not None and limit > 0:
                limit -= (len(attack.poly) - 1) * times
                if limit > 0:
                    if attack.hit_chance == 1:
                        total_dmg = damage_probabilities(attack.poly, times)
                    else:
                        hit_chances = np.fromiter(
                            binomial_distribution(times, attack.hit_chance), sympy.Rational, count=times + 1)
                        total_dmg = (
                            np.row_stack([fix_length(damage_probabilities(attack.poly, i), e_h)
                                          for i in range(times + 1)])
                            * hit_chances[:, np.newaxis]
                        ).sum(axis=0)
                    base = fix_length(poly_mul(base, total_dmg), e_h)
                    continue

            mat = fix_width(poly_mul(q := attack_outcomes(attack, times, e_h), w := base), e_h)
            print(q)
            print(w)
            print(mat.shape)
            array_list.append(mat)
            base = mat[-1]
        damage_rows = np.row_stack(array_list)
        all_damage_rows.append(damage_rows)
        wins.append(np.ediff1d(damage_rows[:, -1], to_begin=damage_rows[0, -1]))
    print(wins)
    print(wins[0].shape, wins[1].shape)
    print(all_damage_rows[0])
    print(all_damage_rows[1])
    print(all_damage_rows[0].shape, all_damage_rows[1].shape)
    return (
        np.sum(wins[0][:, np.newaxis] * all_damage_rows[1], axis=0),
        np.sum(wins[1][:, np.newaxis] * all_damage_rows[0], axis=0),
    )


if __name__ == '__main__':

    a, b = battle_outcomes(
        20, {Attack(np.array(get_combined_dice({2: 1})), sympy.Rational(1, 2), 12): 8},
        18, {Attack(np.array(get_combined_dice({2: 1})), sympy.Rational(1, 2), 12): 7,
             Attack(np.array(get_combined_dice({2: 2})), sympy.Rational(1, 2), 12): 1},
        17, 17
    )
    print(a.astype(float))
    print(b.astype(float))


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
