import sympy

from main import get_combined_dice


x = sympy.symbols('x')


def to_expr_hit_chance(coefficients, hit_chance):
    s = sum(coefficients)
    p = sympy.Poly(sympy.Rational(1, s) * sum(c * x ** i for i, c in enumerate(coefficients)))
    return p * hit_chance + (1 - hit_chance)


def run(moves, hp):
    expr = sympy.Poly(1, x)
    exp = var = 0
    for i, (move, hit) in enumerate(moves, 1):
        expr *= to_expr_hit_chance(move, hit)
        win_chance = sum(expr.all_coeffs()[:-hp])
        exp += i * win_chance
        var += i * win_chance * win_chance
    return exp, var


def main():
    moves = [(get_combined_dice({4: 1}), 1), (get_combined_dice({2: 2}), 1), (get_combined_dice({5: 1}), 1)]
    print(run(moves, 10))


if __name__ == '__main__':
    main()
