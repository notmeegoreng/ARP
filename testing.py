from main import *


def test_battle_mul():
    combatants = [
        Player.from_coeffs(10, get_combined_dice({4: 1}), sympy.Rational(1, 2), 0),
        Player.from_coeffs(487, get_combined_dice({1: 30, 6: 2, 8: 2, 10: 2}), sympy.Rational(19, 20), -1),
        Player.from_coeffs(136, get_combined_dice({1: 4, 8: 2}), sympy.Rational(27, 40), -1),
        Player.from_coeffs(178, get_combined_dice({1: 6, 8: 3}), sympy.Rational(3, 4), 1),
        Player.from_coeffs(207, get_combined_dice({1: 18, 6: 2, 8: 2, 10: 2}), sympy.Rational(4, 5), 0),
        Player.from_coeffs(172, get_combined_dice({1: 18, 6: 2, 8: 2, 10: 2}), sympy.Rational(33, 40), 0)
    ]

    print('''
1: 1d4, 10 hp, 50%, 0
2: 30+2d6+2d8+2d10, 487 hp, 95%, -1
3: 2d8+4, 136 hp, 67.5%, -1
4: 3d8+6 damage, 178 hp, 75%, +1
5: 18+2d6+2d8+2d10, 207 hp, 80%, 0
6: 18+2d6+2d8+2d10, 172 hp, 82.5%, 0
''')

    start = time.perf_counter_ns()
    for p in combatants:
        print(expected_var(p, 100, 200)[1].evalf())
    end = time.perf_counter_ns()

    print(end - start)
    start = time.perf_counter_ns()
    for i, a in enumerate(combatants, 1):
        for j, b in enumerate(combatants, 1):
            if i <= j:
                a_win, b_win = win_chances_initiative(a, b, 200)
                print(f'Combatant {i} v Combatant {j}\n'
                      f'Combatant {i} win: {float(a_win)}\n'
                      f'Combatant {j} win: {float(b_win)}\n')
    end = time.perf_counter_ns()
    print(end - start)


def test_dice():
    player_dice = ({2: 1}, {4: 1}, {6: 1}, {4: 2}, {6: 2}, {10: 1}, {6: 1, 1: 3})
    players = (Player.from_coeffs(1, get_combined_dice(dice), 1, 0) for dice in player_dice)

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


def test_hp():
    start = time.perf_counter_ns()
    p = Player.from_coeffs(1, [1, 1], 1, 0)
    print('1d2 100% hit chance')
    for hp in (10, 100, 1000):
        exp, var = expected_var(p, hp)
        print(f'{hp} HP: E: {float(exp)}, V: {float(var)}')
    hp = 2
    while hp <= 512:
        exp, var = expected_var(p, hp)
        print(f'{hp} HP: E: {float(exp)}, V: {float(var)}')
        hp *= 2
    end = time.perf_counter_ns()
    print(end - start)

    # note: takes so long I have not done it yet
    start = time.perf_counter_ns()
    exp, var = expected_var(p, 10000, exact=False)
    print(f'{hp} HP: E: {float(exp)}, V: {float(var)}')
    end = time.perf_counter_ns()
    print(end - start)


def test_battle():
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


if __name__ == '__main__':
    test_battle_mul()
