"""
This child of the "Attack" class implements the Hill Climbing heuristic.
This is the simplest heuristic; it simply chooses the best sample in the neighborhood.
"""

import sys
import json
import numpy as np
from tqdm import tqdm
sys.path.append("../master")

# Importing necessary libraries
from attacks.header import *
from attacks.Attack import Attack
from neighborhoods.Neighborhood import Neighborhood
from neighborhoods.Radar import Radar

# Load parameters from JSON
with open('../master/master_params.json', 'r') as file:
    params = json.load(file)

# Mapping from strings to actual functions/classes
object_map = {
    "L2_norm": L2_norm,
    "Radar": Radar,
    # Add other functions/classes as needed
}

# Extracting the default parameters for HillClimbing attack
params_attack = params["attack"]["HillClimbing"]

# Defining the HillClimbing class
class HillClimbing(Attack):
    """
    Child of the Attack class that implements the Hill Climbing heuristic.
    """
    def __init__(self, estimator=None, verbose:int=1):
        """
        Constructor for the HillClimbing class.
        """
        super().__init__(estimator, verbose)
        # Load default run parameters from the JSON file
        self.run_params = params_attack.copy()
        # Map string representations to actual objects
        self.run_params['cost_function'] = object_map[self.run_params['cost_function']]
        self.run_params['neighborhood'] = object_map[self.run_params['neighborhood']]
        self.verbose = verbose

    def verbose_message(self, message):
        if self.verbose > 1:
            print(message)

    def run(self, input, **kwargs):
        """
        Run the HillClimbing attack.

        Parameters:
            input: The input data.
            **kwargs: Optional parameters that override defaults in self.run_params.
        """
        # Update run_params with any kwargs provided
        params = self.run_params.copy()
        params.update(kwargs)

        # Map any overridden string parameters to actual objects
        if 'cost_function' in kwargs:
            if isinstance(kwargs['cost_function'], str):
                params['cost_function'] = object_map[kwargs['cost_function']]
            else:
                params['cost_function'] = kwargs['cost_function']

        if 'neighborhood' in kwargs:
            if isinstance(kwargs['neighborhood'], str):
                params['neighborhood'] = object_map[kwargs['neighborhood']]
            else:
                params['neighborhood'] = kwargs['neighborhood']

        # Extract parameters
        cost_function = params['cost_function']
        targeted = params['targeted']
        specific_class = params['specific_class']
        neighborhood = params['neighborhood']
        static_perturbation_factor = params['static_perturbation_factor']
        dynamic_perturbation_factor = params['dynamic_perturbation_factor']
        max_iter = params['max_iter']
        patience = params['patience']
        accept_improving_candidates_only = params['accept_improving_candidates_only']
        raise_dynamic_perturbation_after_non_improving_candidate = params['raise_dynamic_perturbation_after_non_improving_candidate']
        inflation_vector_max_perturbation = params['inflation_vector_max_perturbation']
        enable_negative_inflation_vector = params['enable_negative_inflation_vector']

        # Initialize neighborhood instance if needed
        if isinstance(neighborhood, type):
            neighborhood = neighborhood()
        
        # Checking the constraints format
        neighborhood.check_constraints_format(len(input))
        self.verbose_message("The constraints have the correct format.")

        # Check initial class
        y_initial = self.estimator.predict([input])[0]
        self.verbose_message(f"Initial label is {y_initial}.")

        # Initialize heuristic history
        self.heuristic_history.append([])
        size_in_bytes = sys.getsizeof(self.heuristic_history)
        self.verbose_message(f"Memory allocated to heuristic history: {size_in_bytes} bytes.")

        # Initializing the best sample and its cost
        sample = input
        best_sample = None
        best_cost = np.inf  # We want to minimize the cost
        total_queries = 0
        dyna = dynamic_perturbation_factor

        # Set patience counter
        p = patience

        # Running the Hill Climbing heuristic
        for _ in tqdm(range(max_iter), disable=(not self.verbose)):
            # Generating a sample
            sample, queries = neighborhood.generate_valid(
                sample,
                estimator=self.estimator,
                y=y_initial,
                is_targeted_attack=targeted,
                targeted_class=specific_class,
                static_perturbation_factor=static_perturbation_factor,
                dynamic_perturbation_factor=dyna,
                enable_negative_inflation_values=enable_negative_inflation_vector,
                inflation_vector_max_perturbation=inflation_vector_max_perturbation,
            )
            total_queries += queries

            # If sample is a list, it means that the neighborhood generated several candidates.
            # In this case, according to the Hill Climbing heuristic, we choose the best candidate.
            if isinstance(sample, list):
                # sample = the candidate which leads to the minimum cost
                sample = min(sample, key=lambda x: cost_function(x, input))

            # Computing the cost
            cost = cost_function(sample, input)

            if best_sample is not None:
                self.heuristic_history[-1].append([best_sample, best_cost])

            # Checking if the cost is better
            if cost < best_cost:
                dyna = dynamic_perturbation_factor
                p = patience
                best_sample = sample.copy()
                best_cost = cost
            elif p > 0:
                p -= 1
                if raise_dynamic_perturbation_after_non_improving_candidate:
                    dyna *= dynamic_perturbation_factor
                if accept_improving_candidates_only and best_sample is not None:
                    sample = best_sample  # Rollback to best sample
            else:
                break

        if best_cost == np.inf:
            raise ValueError("Impossible to find a sample satisfying constraints and misclassification.")

        # Last check
        total_queries += 1
        prediction = self.estimator.predict([best_sample])[0]
        if (targeted and prediction != specific_class) or (not targeted and prediction == y_initial):
            raise ValueError("This is embarrassing... The final sample is not adversarial!")
        
        return best_sample, best_cost, total_queries 