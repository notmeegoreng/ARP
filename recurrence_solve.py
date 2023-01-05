import sympy

import main

y = sympy.Function('y')
x = sympy.symbols('x', integer=True)


# 1d2 ([1, 1]) and 1d3 ([1, 1, 1]) can be done pretty quickly,
# but it seems like it cannot handle 1d4 ([1, 1, 1, 1])

def recurrence_solve(damage):
    """
    :param damage: a list of relative chances of dealing damage equal to the index (one-indexed)
    :return: Recurrence equation for a function to calculate E(X) given HP.
    Note: might not work, but it will try.
    It appears not to be able to handle [1, 1, 1, 1], but [1, 1, 1] works.
    """
    s = sum(damage)
    f = sympy.sympify('+'.join(f'(y(x - {i}) + 1)*{c}' for i, c in enumerate(damage, 1)),
                      {'y': y, 'x': x}) / s - y(x)
    print(f)
    initial = {y(-1): 0, y(0): 0}
    for i in range(1, len(damage) - 1):
        initial[y(i)] = main.expected_var(main.Player.from_coeffs(0, damage, 1, 0), i)[0]

    print(initial)
    expr = sympy.rsolve(f, y(x), initial)
    print(expr)
    for i in (0, 1, 10, 20, 50, 100, 1000):
        print(expr.subs(x, i).evalf(chop=True))


if __name__ == '__main__':
    recurrence_solve([1, 1])
