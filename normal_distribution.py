from scipy.stats import norm

from linear_approximation import linear_predict
from recurrence import recurrence_method


def approx_win(diff: float, total_var: float):
    return norm.cdf(diff * total_var ** -0.5)


def approx_win_linear(damage_0, hp_0, damage_1, hp_1, total_var):
    diff = linear_predict(damage_0, hp_1) - linear_predict(damage_1, hp_0)
    return approx_win(diff, total_var)


def approx_win_recurrence(damage_0, hp_0, damage_1, hp_1, total_var):
    diff = recurrence_method(damage_0, hp_1)[-1] - recurrence_method(damage_1, hp_0)[-1]
    return approx_win(diff, total_var)
