import sympy

import main

y = sympy.Function('y')
x = sympy.symbols('x', integer=True)

# 1d2 ([1, 1]) and 1d3 ([1, 1, 1]) can be done pretty quickly,
# but it seems like it cannot handle 1d4 ([1, 1, 1, 1])

coeffs = [1, 1]  # damage coefficients
s = sum(coeffs)
f = sympy.sympify('+'.join(f'(y(x - {i}) + 1)*{c}' for i, c in enumerate(coeffs, 1)),
                  {'y': y, 'x': x}) / s - y(x)
print(f)
initial = {y(-1): 0, y(0): 0}
for i in range(1, len(coeffs) - 1):
    initial[y(i)] = main.expected_var(main.Player.from_coeffs(0, coeffs, 1, 0), i)[0]

print(initial)
expr = sympy.rsolve(f, y(x), initial)
print(expr)
for i in (0, 1, 10, 20, 50, 100, 1000):
    print(expr.subs(x, i).evalf(chop=True))
