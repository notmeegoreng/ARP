import time

import sympy

from main import Player, get_combined_dice, win_chances, win_chances_initiative, expected_var
from recurrence import recurrence_method
from linear_approximation import linear_predict

p0 = Player.from_coeffs(20, get_combined_dice({2: 2}), sympy.Rational(2, 3), 0)
p1 = Player.from_coeffs(15, get_combined_dice({2: 3}), sympy.Rational(1, 1), 0)

# ex0, var0 = expected_var(p0, p1.hp, 100)

start = time.perf_counter_ns()
print(linear_predict(get_combined_dice({2: 2}), p1.hp))
end = time.perf_counter_ns()
print('A time:', end - start)

start = time.perf_counter_ns()
print(recurrence_method(get_combined_dice({2: 2}), p1.hp)[-1])
end = time.perf_counter_ns()
print('B time:', end - start)

start = time.perf_counter_ns()
print(*map(float, win_chances_initiative(p0, p1, 1000)))
end = time.perf_counter_ns()


print('C time:', end - start)

#
# print(1 - sum(win_chances(p0, p1, 1000)))

# damage spells list:
dsl = [[0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 1 / 2, 0, 1 / 2]]
# list of which spell is chosen, at which hp, ie if at 5 hp, choicelist[5] returns the optimal spell at 5 hp
choicelist = [0]
# turn list, a list of roughly how many turns expected to kill something if thing is at n hp
hplist = [0]
n = 7
# start the for loop, with hp =1, thus calculating for hp list, and helping us to build choices
hp = 1
for count in range(1000):
    # for each one of the spells,
    minimum = 100000
    minspell = 1000
    for count2 in range(len(dsl)):
        expect = 1

        # for each one of the damages in dsl, attempt to calculate the number of turns itll take if it goes does n damage for rest of combat
        for count3 in range(len(dsl[count2])):
            if (count3 == 0):
                expect += dsl[count2][count3] / (1 - dsl[count2][count3])
            if (count3 >= hp):
                expect += 0
            else:
                expect += dsl[count2][count3] * hplist[hp - 1 - count3]
        if (minimum > expect):
            minimum = expect
            minspell = count2
    hp += 1

    hplist.append(minimum)
    choicelist.append(minspell)
print(hplist)
print(choicelist)