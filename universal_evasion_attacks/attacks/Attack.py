"""
Abstract class for all attacks to come.
- All attacks are based on one heuristic strategy.
- All attacks are based on a neighborhood strategy, which is independant from the heuristic.
- All attacks contain must contain a "set_estimator" method for saving the predictive model to attack.
- All attacks contain a "run" method for executing the attack with a set of parameters depending on the heuristic and neighborhood strategies.
    - The "run" method always takes 1 input of the form of a list of numerical values.
    - The "run" method always returns a list of numerical values (final answer)
    - It is necessary to specify whether the attack is targeted (estimator(input) = specific_class) or untargeted (estimator(input) != specific_class).
- A verbose option must be available for the user to see the attack in action.
- Tqdm bars keep track of the iterative process.
- Attacks should (optionally) keep track of the heuristic history, that said, a list of the form [{"1": **sample1, "2": **sample2, ...}, ...].
"""

# Importing necessary libraries
from attacks.header import *

# Defining the abstract class for all attacks
class Attack:
    """
    Abstract class for all heuristic attacks to come.
    """
    def __init__(self, estimator, verbose:int=1):
        """
        Constructor for the Attack class.
        """
        self.estimator = estimator
        self.verbose = verbose # 0: silence, 1: only the tqdm bars, 2: all the details
        self.heuristic_history = []

    def set_estimator(self, estimator):
        """
        Method for setting the estimator to attack.
        """
        self.estimator = estimator

    def run(self, input):
        """
        Asbtract method (should be defined in the child class).
        """
        pass