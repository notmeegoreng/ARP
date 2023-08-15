import dataclasses
import itertools
from collections.abc import Iterator

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
    attacks: dict[Attack, int]
    initiative_mod: int

    def __post_init__(self):
        # sort attacks by expected damage
        object.__setattr__(
            self, 'attacks',
            dict(sorted(self.attacks.items(), key=lambda item: item[0].expected_dmg(), reverse=True)))


damage_cache = {}


def damage_probabilities(base: np.ndarray, hits: int) -> np.ndarray:
    if hits < 0:
        raise ValueError('hits >= 0!')
    if hits == 0:
        return np.ones(1, dtype=gmpy2.mpq)
    h = hash(base.tobytes())
    if h not in damage_cache:
        damage_cache[h] = [base]
    for _ in range(len(damage_cache[h]), hits):
        damage_cache[h].append(poly_mul(damage_cache[h][-1], base))
    return damage_cache[h][hits - 1]


def win_by(poly: np.ndarray, n: int) -> gmpy2.mpq:
    if len(poly) == 1:
        return gmpy2.mpq()
    return poly[n:].sum()


def win_on(poly: np.ndarray, n: int, t: int) -> gmpy2.mpq:
    return win_by(damage_probabilities(poly, t), n) - win_by(damage_probabilities(poly, t - 1), n)


def get_attacks_greedy(p: Player, turns: int) -> dict[Attack, int]:
    # greedy algo
    result = {}
    for att, uses in p.attacks.items():
        if uses == 0 or turns <= uses:
            result[att] = turns
            return result
        else:
            result[att] = uses
            turns -= uses

    raise ValueError('Not enough attacks to use for that many turns!')


def get_attacks(p: Player, enemy_hp: int, turns: int) -> Iterator[Attack | None]:
    # greedy algo
    for att, uses in p.attacks.items():
        if turns <= uses:
            enemy_hp // att.max_dmg()
            for _ in range(uses):
                yield att
            if turns == uses:
                return
            turns -= uses
        else:
            for _ in range(turns):
                yield att
            return
    raise ValueError('Not enough attacks to use for that many turns!')


def combine_attacks(attacks: dict[Attack, int]) -> np.ndarray:
    poly = np.ones(1, dtype=gmpy2.mpq)
    for attack, times in attacks.items():
        poly = poly_mul(poly, damage_probabilities(attack.poly, times))
    return poly


def win_by_attacks(attacks: dict[Attack, int], hp: int):
    return win_by(combine_attacks(attacks), hp)


def get_min_turns(p: Player, hp: int):
    turns = 0
    for att, uses in p.attacks.items():
        max_dmg = len(att.poly)
        if uses != 0 and max_dmg * uses <= hp:
            turns += uses
            hp -= max_dmg * uses
            if hp == 0:
                return turns
        else:
            return turns + (hp - 1) // max_dmg + 1


def win_by_array(attacks: dict[Attack, int], enemy_hp: int, max_turns: int | None = None, *,
                 initial: np.ndarray | None = None) -> np.ndarray:
    if initial is None:
        poly = np.ones(1, dtype=gmpy2.mpq)
    else:
        poly = initial

    # when attacks have a hit chance, the battle could theoretically go on forever
    # we calculate up to a given number of turns
    wins_by = np.empty(max_turns or sum(attacks.values()), dtype=gmpy2.mpq)
    t = 0
    for att, count in attacks.items():
        for _ in range(count):
            poly = poly_mul(poly, att.poly)
            wins_by[t] = win_by(poly, enemy_hp)
            t += 1
    return wins_by


def win_arrays(p0: Player, p1: Player, turns: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    min_turns = min(get_min_turns(p0, p1.hp), get_min_turns(p1, p0.hp))
    if turns < min_turns:
        raise ValueError('Not enough turns - confirmed draw')

    wins_by0 = win_by_array(get_attacks_greedy(p0, turns), p1.hp, turns)
    wins_by1 = win_by_array(get_attacks_greedy(p1, turns), p0.hp, turns)

    # chance of winning on turn n = chance of winning by turn n - chance of winning by turn n-1
    wins0 = np.ediff1d(wins_by0, to_begin=wins_by0[0])
    wins1 = np.ediff1d(wins_by1, to_begin=wins_by1[0])

    # chance of a player winning each turn:
    # chance they defeat other on that turn * chance that they have not been defeated
    # chance of simultaneous elimination each turn: chance each defeats other multiplied together
    return wins0 * (1 - wins_by1), wins1 * (1 - wins_by0), wins0 * wins1


def all_win_arrays(
        initial: np.ndarray, enemy_hp: int,
        attacks_remaining: dict[Attack, int], turns_remaining: int) -> list[tuple[dict[Attack, int], np.ndarray]]:
    num_attacks_left = sum(attacks_remaining.values())
    if num_attacks_left < turns_remaining:
        raise ValueError('not enough attacks')
    elif num_attacks_left == turns_remaining:
        return [(attacks_remaining, win_by_array(attacks_remaining, enemy_hp, turns_remaining, initial=initial))]
    arrays = []
    for product in itertools.product(*(
            range(uses) if uses else range(turns_remaining) for uses in attacks_remaining.values()
    )):
        if sum(product) == turns_remaining:
            attacks = dict(zip(attacks_remaining.keys(), product))
            win = win_by_array(attacks, enemy_hp, turns_remaining)
            # sort out superseded
            for array in arrays[:]:
                if (array <= win).all():
                    arrays.remove(array)
                    print('c0')
                elif (array >= win).all():
                    print('c1')
                    break
            else:  # no break
                arrays.append((attacks, win))
    return arrays


def greedy_until_hp(p: Player) -> int | None:
    """What enemy HP we can use greedy algorithm until"""
    if len(p.attacks) == 1:
        return None
    if len(p.attacks) == 2:
        atts = iter(p.attacks.keys())
        return next(atts).max_dmg() * next(atts).max_dmg()


def greedy_until(p: Player, enemy_hp: int) -> dict[Attack, int] | None:
    min_hp = greedy_until_hp(p)
    if min_hp is None:
        return None
    if min_hp < enemy_hp:
        return {}
    enemy_hp -= min_hp
    result = {}
    for att, uses in p.attacks.items():
        dmg = att.max_dmg()
        if uses != 0 and enemy_hp // dmg >= uses:
            enemy_hp -= dmg * uses
            result[att] = uses
            if not enemy_hp:
                return result
        else:
            result[att] = (enemy_hp - 1) // dmg + 1
            return result
    return result


def exhaustive_single(p: Player, enemy_hp: int, turns: int) -> list[tuple[dict[Attack, int], np.ndarray]]:
    initial_attacks = greedy_until(p, enemy_hp)
    if initial_attacks is None:
        return [(atts := get_attacks_greedy(p, turns), win_by_array(atts, enemy_hp, turns))]

    initial = combine_attacks(initial_attacks)
    attacks_remaining = {att: uses - used for (att, uses), used in zip(p.attacks.items(), initial_attacks.values())}
    return all_win_arrays(initial, enemy_hp, attacks_remaining, sum(attacks_remaining.values()))


def win_compare(w0: np.ndarray | None, w1: np.ndarray) -> bool:
    """True if w1 is considered larger, otherwise false"""
    if w0 is None:
        return True
    for c0, c1 in zip(w0, w1):
        if c0 < c1:
            return True
        elif c0 > c1:
            return False


def best_exhaustive_single(p: Player, enemy_hp: int, turns: int):
    max_win = None
    max_win_att = None
    for attacks, win in exhaustive_single(p, enemy_hp, turns):
        if win_compare(max_win, win):
            max_win = win
            max_win_att = attacks
    return max_win_att


def exhaustive(p0: Player, p1: Player, turns: int):
    attacks0 = best_exhaustive_single(p0, p1.hp, turns)
    attacks1 = best_exhaustive_single(p1, p0.hp, turns)
    return attacks0, attacks1


def fix_length(dmg: np.ndarray, length: int):
    """Fixes the length of 1D arrays"""
    if len(dmg) < length:
        return np.pad(dmg, (0, length - len(dmg)))
    elif len(dmg) > length:
        excess = dmg[length:].sum()
        dmg = dmg[:length]
        dmg[-1] += excess
        return dmg
    return dmg


def fix_width(mat: np.ndarray, width: int) -> np.ndarray:
    """Fixes the width of 2D arrays"""
    curr = mat.shape[-1]
    if curr < width:
        return np.pad(mat, ((0, 0), (0, width - curr)))
    elif curr > width:
        excess = mat[:, width:].sum(axis=1)
        mat = mat[:, :width]
        mat[:, -1] += excess
        return mat
    return mat


def attack_outcomes(attack: Attack, uses: int, width: int):
    # omit first row of all zeros for eff
    rows = [fix_length(attack.poly, width)]
    for _ in range(uses - 1):
        rows.append(fix_length(poly_mul(rows[-1], attack.poly), width))
    return np.row_stack(rows)


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
        base = np.ones(1, dtype=gmpy2.mpq)
        array_list = [np.zeros(e_h, dtype=gmpy2.mpq)]
        for attack, times in attacks.items():
            attack_mat = fix_width(poly_mul(q := attack_outcomes(attack, times, e_h), w := base), e_h)
            print(q)
            print(w)
            print(attack_mat.shape)
            array_list.append(attack_mat)
            base = attack_mat[-1]
        damage_rows = np.row_stack(array_list)
        all_damage_rows.append(damage_rows)
        wins.append(np.ediff1d(damage_rows[:, -1], to_begin=damage_rows[0, -1]))
    print(wins)
    print(wins[0].shape, wins[1].shape)
    print(all_damage_rows[0])
    print(all_damage_rows[1])
    print(all_damage_rows[0].shape, all_damage_rows[1].shape)
    print(type(all_damage_rows[1][-1, -1]))
    return (
        np.sum(wins[0][:, np.newaxis] * all_damage_rows[1], axis=0),
        np.sum(wins[1][:, np.newaxis] * all_damage_rows[0], axis=0),
    )


def win_chances(p0: Player, p1: Player, turns: int) -> tuple[gmpy2.mpq, gmpy2.mpq, gmpy2.mpq]:
    """
    Given two players, returns chances of the first player winning, the second player winning,
    or a draw (simultaneous elimination or no elimination).
    """
    w0, w1, d = win_arrays(p0, p1, turns)
    return np.sum(w0), np.sum(w1), np.sum(d)  # type: ignore


def expected_var(p: Player, hp: int, turns: int) -> tuple[gmpy2.mpq, gmpy2.mpq]:
    min_turns = get_min_turns(p, hp)
    if min_turns > turns:
        raise ValueError('Not enough turns - confirmed draw')

    wins_by = win_by_array(get_attacks_greedy(p), hp, turns)

    wins = np.ediff1d(wins_by, to_begin=wins_by[0])
    exp = np.arange(1, turns + 1) * wins
    return np.sum(exp), np.sum(exp * wins)  # type: ignore


def main():
    sw = 5
    if sw == 0:
        print(w := win_chances(
            Player(3, {Attack.from_dice({3: 1}, gmpy2.mpq(1, 2)): 0}, 1),
            Player(3, {Attack.from_dice({3: 1}, gmpy2.mpq(1, 2)): 0}, 1),
            2
        ))
        print([float(x) for x in w])
    elif sw == 1:
        a, b = battle_outcomes(
            10, {Attack.from_dice({2: 1}, gmpy2.mpq(1, 2)): 8},
            10, {Attack.from_dice({2: 1}, gmpy2.mpq(1, 2)): 7,
                 Attack.from_dice({2: 2}, gmpy2.mpq(1, 2)): 1},
            17, 17
        )
        print(a.astype(float))
        print(b.astype(float))
    elif sw == 2:
        exp, var = expected_var(Player(3, {Attack.from_dice({2: 2}, gmpy2.mpq(1, 1)): 0}, 1), 20, 100)
        print(exp, var)
        print(float(exp), float(var))
    elif sw == 3:
        print(exhaustive(
            Player(10, {Attack.from_dice({2: 1}, gmpy2.mpq(1, 2)): 0}, 1),
            Player(8, {Attack.from_dice({2: 1}, gmpy2.mpq(1, 2)): 0,
                       Attack.from_dice({2: 2}, gmpy2.mpq(1, 2)): 2}, 1),
            100
        ))


if __name__ == '__main__':
    main()
