# Dungeons and Dragons Inspired Cantrip Knockout Prediction: Error-minimising Numerical Investigative System 

This repository was made as part of this research project. It contains all the code that was used in our investigation. Our abstract is shown below.


>Dungeons and Dragons (DnD) is a common turn-based role playing game, primarily based on combat. Attempts to accurately and quickly calculate the likelihood of winning combat has been difficult, due to three factors: The large variance in damage dealt during a hit, initiative rolls to determine who goes first, as well as probability of a hit.  Current methods of prediction lie in a few forms: a program is used to simulate DnD combat, and runs the battle a large number of times, thus giving results on likelihoods. This method is inefficient due to the high amount of time it spends, as a large number of fights must be modelled.
> 
>Another method commonly used is Challenge Ratings (CR). While CR ratings are useful, they have a strong limitation: they are very subjective. A million lions would lose against a sun dragon with CR 20 even though it seems like they would win based on CR alone.
> 
>Our methods aim to be simple ways to predict DnD fights with probability generating functions. We developed a computer program which is capable of modelling cantrip knockouts - a type of DnD fight where the combatants repeatedly use reusable spells to attack - with high accuracy and speed. We dealt with the three problems mentioned above separately. Probability generating functions can handle the variance in damage dealt, the program can handle the problem of likelihood to hit through calculation, and the pattern in 20 sided dice can help find chances of winning initiative rolls. We also developed some faster mathematical methods that model cantrip knockouts, which retain high accuracy with less effort. This involves looking at E(X), the expected number of turns for a combatant to deal enough damage and defeat their opponent.

The three strategies are implemented in:
* Linear Approximation - `linear_approximation.py`
* Recurrence Method - `recurrence.py`
    * For simple cases (where the damage probability generating function is 3 terms long at most), the reccurence relation can be solved for a single function - `reccurence_solve.py`
* Computer Simulation - `main.py`
    
