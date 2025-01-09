"""
Abstract class for all neighborhood classes to come.
- All neighborhoods take as input a dictionary of constrains:
    { "equality": ["x[0] = int(x[0])", ...],
      "inequality": ["x[1] > x[2]**2", ...],
      "clip_min": [0, None, ...], # minimum values for each feature, this list is the size of the input x
      "clip_max": [1, None, ...], # maximum values for each feature
      "categorical": [None, [0, 1], ...], # categorical features
    }
- All neighborhoods must contain a "generate" method for generating a neighborhood of a given input.
- All neighborhoods use the original input x to generate one or more samples.
"""

import re

# Defining the abstract class for all neighborhoods
class Neighborhood:
    """
    Abstract class for all neighborhood classes to come.
    """
    def __init__(self, constraints:dict, verbose:int=1):
        """
        Constructor for the Neighborhood class.
        """
        self.constraints = constraints
        self.verbose = verbose

    def generate(self, x) -> list:
        """
        Abstract method (should be defined in the child class).
        """
        pass

    def generate_valid(self, x, estimator, y=None, max_trials=10000, is_targeted_attack=False,
                        targeted_class=None, static_perturbation_factor=1e-6,
                          dynamic_perturbation_factor=1.2, inflation_vector_max_perturbation=2,
                          enable_negative_inflation_values=False, initial_perturbation_vector=None):
        """
        Abstract method to generate a valid sample that satisfies the constraints of the evasion attack.
        """
        pass

    def extract_constraints(self):
        return self.constraints["equality"], self.constraints["inequality"], self.constraints["categorical"], self.constraints["clip_min"], self.constraints["clip_max"]

    def check_constraints_format(self, input_size:int):
        """
        Method for checking if the constraints dictionary has the correct format.
        """
        #if self.verbose > 0:
        #    print("Checking constraints format... (constraints coherence is not checked)")

        # Extracting the constraints
        try:
            equalities, inequalities, categorical, clip_min, clip_max = self.extract_constraints()
        except KeyError:
            raise ValueError("Constraints dictionary must contain the keys 'equality', 'inequality', 'categorical', 'clip_min', and 'clip_max'.")
        
        if not isinstance(equalities, list) or not isinstance(inequalities, list) or not isinstance(categorical, list) or not isinstance(clip_min, list) or not isinstance(clip_max, list):
            raise ValueError("All constraints in the constraints dictionnary must be lists.")

        # 1st check: equality constraints must be of the form "x[INDEX] = FUNCTION(x, ...)" so "x[1] = 1/x[0]**2+7"
        # or "x[2] = int(x[2])" are valid, but "x[1] > 1" is not.
        for equality in equalities:
            if not re.match(r"x\[\d+\] = .+", equality):
                raise ValueError(f"Equality constraint {equality} does not have the correct format.")
            
        # 2nd check: inequality constraints must be of the form "x[INDEX] > FUNCTION(x, ...)"
        # it is important that no space appears in the right side of the inequality sign.
        # So "x[1] > 1+x[0]" is fine, but "x[1] > 1 + x[0]" is not.
        for inequality in inequalities:
            if not re.match(r"x\[\d+\] > .+", inequality):
                raise ValueError(f"Inequality constraint {inequality} does not have the correct format.")
            
        # 3rd check: clip_min and clip_max must be lists of the same size as the input x.
        if len(clip_min) != input_size or len(clip_max) != input_size:
            raise ValueError("clip_min and clip_max must have the same size as the input x.")
        # Also, a min clip value cannot be greater than a max clip value.
        for mini, maxi in zip(clip_min, clip_max):
            if mini is not None and maxi is not None and mini > maxi:
                raise ValueError("Minimum clip value cannot be greater than maximum clip value.")
        
        # 4th check: categorical must be a list of the same size as the input x, where each element is either None or a list.
        if len(categorical) != input_size:
            raise ValueError("Categorical constraints must have the same size as the input x.")
        for category in categorical:
            if category is not None and not isinstance(category, list):
                raise ValueError("Categorical constraints must be lists or None.")
            
        #if self.verbose > 0:
        #    print("Constraints format is correct.")