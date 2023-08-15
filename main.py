import dataclasses
from collections.abc import Sequence

import gmpy2
import numpy as np


gmpy2.get_context().rational_division = True


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


def poly_pow(poly: np.ndarray, n: int) -> np.ndarray:
    r = poly
    n -= 1
    while True:
        n, m = n // 2, n

        if m & 1:
            r = poly_mul(poly, r)

        if not n:
            break

        r = poly_mul(r, r)
    return r


@dataclasses.dataclass(frozen=True)
class Attack:
    poly: np.ndarray

    @classmethod
    def from_dice(cls, dice: dict[int, int], hit_chance: gmpy2.mpq):
        total = np.ones(1, dtype=gmpy2.mpq)
        for sides, n in dice.items():
            # mul the pgf of this dice type
            dice = np.ones(sides + 1, dtype=gmpy2.mpq)
            dice[0] = 0
            total = poly_mul(poly_pow(dice, n), total)
        total *= hit_chance / total.sum()
        total[0] = 1 - hit_chance
        return cls(total)

    def expected_dmg(self):
        return (self.poly * np.arange(len(self.poly))).sum()

    def max_dmg(self):
        return len(self.poly) - 1

    def __hash__(self):
        return hash((self.poly.tobytes()))


@dataclasses.dataclass(frozen=True)
class Player:
    hp: int
    attacks: Sequence[Attack]
    initiative_mod: int

    def __post_init__(self):
        # sort attacks by expected damage
        object.__setattr__(
            self, 'attacks',
            dict(sorted(self.attacks.items(), key=lambda item: item[0].expected_dmg(), reverse=True)))  # type: ignore


def attack_choices(p: Player, up_to: int):
    """Finds what attacks to use at each enemy HP in order to defeat them as fast as possible"""
    choice_list = [None]
    # turn list, a list of roughly how many turns expected to kill something if thing is at n hp
    hp_list = [0]
    # start the for loop, with hp = 1, thus calculating for hp list, and helping us to build choices
    enemy_hp = 1
    for _ in range(up_to):
        # for each one of the spells,
        minimum = 100000
        min_spell = 1000
        for attack in p.attacks:
            expect = 1

            # for each one of the damages in dsl, attempt to calculate the number of turns it'll take
            # if it goes does n damage for rest of combat
            for i in range(min(len(attack.poly), enemy_hp)):
                if i == 0:
                    expect += attack.poly[0] / (1 - attack.poly[0])
                expect += attack.poly[i] * hp_list[enemy_hp - i - 1]
            if minimum > expect:
                minimum = expect
                min_spell = attack
        enemy_hp += 1

        hp_list.append(minimum)
        choice_list.append(min_spell)

    return choice_list


split_cache = {}


def split_with_choices(choices: Sequence[Attack], enemy_hp: int, turns_remaining: int):
    if enemy_hp <= 0:
        return 1
    if turns_remaining == 0:
        return 0
    c_hash = hash(choices)
    if chance := split_cache.get((c_hash, enemy_hp, turns_remaining)):
        return chance

    poly = choices[enemy_hp].poly
    chance = np.sum(poly * np.fromiter(
        (split_with_choices(choices, hp, turns_remaining - 1) for hp in range(enemy_hp, enemy_hp - len(poly) - 1, -1)),
        gmpy2.mpq, len(poly)
    ))
    split_cache[(c_hash, enemy_hp, turns_remaining)] = chance
    return chance


def split(p: Player, enemy_hp: int, turns: int) -> np.ndarray:
    """
    Finds the chance of winning on a certain number of turns by splitting into branches with recursion.
    Attacks chosen by `attack_choice`
    """
    choices = attack_choices(p, enemy_hp)
    return split_with_choices(tuple(choices), enemy_hp, turns)


def win_array(p: Player, enemy_hp: int, max_turns: int):
    choices = attack_choices(p, enemy_hp)
    return np.fromiter((split_with_choices(choices, enemy_hp, t) for t in range(max_turns)),
                       dtype=gmpy2.mpq, count=max_turns)


def win_arrays(p0: Player, p1: Player, turns: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    wins_by0 = win_array(p0, p1.hp, turns)
    wins_by1 = win_array(p1, p0.hp, turns)

    # chance of winning on turn n = chance of winning by turn n - chance of winning by turn n-1
    wins0 = np.ediff1d(wins_by0, to_begin=wins_by0[0])
    wins1 = np.ediff1d(wins_by1, to_begin=wins_by1[0])

    # chance of a player winning each turn:
    # chance they defeat other on that turn * chance that they have not been defeated
    # chance of simultaneous elimination each turn: chance each defeats other multiplied together
    return wins0 * (1 - wins_by1), wins1 * (1 - wins_by0), wins0 * wins1


def win_chances(p0: Player, p1: Player, turns: int,) -> tuple[gmpy2.mpq, gmpy2.mpq, gmpy2.mpq]:
    """
    Given two players, returns chances of the first player winning, the second player winning,
    or a draw (simultaneous elimination or no elimination).
    """
    w0, w1, d = win_arrays(p0, p1, turns)
    return np.sum(w0), np.sum(w1), np.sum(d)  # type: ignore


def win_chances_initiative(a: Player, b: Player, turns: int) -> tuple[gmpy2.mpq, gmpy2.mpq]:
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
        ipb = gmpy2.mpq((20 - diff) * (19 - diff), 380 + diff) / 2
        ipa = 1 - ipb
    else:
        ipa = gmpy2.mpq((20 + diff) * (19 + diff), 380 - diff) / 2
        ipb = 1 - ipa

    # first to attack wins in case of simultaneous elimination
    a_w, b_w, d = win_chances(a, b, turns)
    return a_w + ipa * d, b_w + ipb * d