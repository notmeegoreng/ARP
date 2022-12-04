import sympy


def run(coefficients, n):
    x = sympy.symbols('x')
    s = sum(coefficients)
    expr = sympy.Rational(1, s) * sum(c * x ** i for i, c in enumerate(coefficients, 1))
    p = sympy.Poly(expr ** 3)
    p *= expr
    print(p, type(p))


run([1, 2, 5], 1)
