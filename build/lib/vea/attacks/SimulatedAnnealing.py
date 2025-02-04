"""
This child of the "Attack" class implements the Simulated Annealing heuristic.
It uses a probabilistic acceptance criterion to escape local minima.
"""

from vea.config import params
from vea import L2_norm, Balloon, Flower, Lightning, random, tqdm, np, Attack

# Mapping from strings to actual functions/classes
object_map = {
    "L2_norm": L2_norm,
    "Balloon": Balloon,
    "Flower": Flower,
    "Lightning": Lightning
}

# Extracting the default parameters for SimulatedAnnealing attack
params_attack = params["attack"]["SimulatedAnnealing"]

# Defining the SimulatedAnnealing class
class SimulatedAnnealing(Attack):
    """
    Child of the Attack class that implements the Simulated Annealing heuristic.
    """
    def __init__(self, estimator=None, verbose:int=1):
        """
        Constructor for the SimulatedAnnealing class.
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
        Method for executing the Simulated Annealing heuristic.

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
        inflation_vector_max_perturbation = params['inflation_vector_max_perturbation']
        enable_negative_inflation_vector = params['enable_negative_inflation_vector']
        raise_dynamic_perturbation_after_non_improving_candidate = params['raise_dynamic_perturbation_after_non_improving_candidate']
        initial_temperature = params['initial_temperature']
        final_temperature = params['final_temperature']
        cooling_rate = params['cooling_rate']

        # Initialize neighborhood instance if needed
        if isinstance(neighborhood, type):
            neighborhood = neighborhood()

        # Checking the constraints format
        neighborhood.check_constraints_format(len(input))
        self.verbose_message("The constraints have the correct format.")

        # Check initial class
        y_initial = self.estimator.predict([input])[0]
        self.verbose_message(f"Initial label is {y_initial}.")

        # Initializing the current sample, best sample, and their costs
        current_sample = input
        best_sample = None
        current_cost = np.inf
        best_cost = current_cost
        total_queries = 0

        # Initialize temperature
        T = initial_temperature

        # Initialize history (single agent)
        heuristic_history = []

        # Initialize patience
        p = patience

        # Initialize dynamic perturbation factor
        dyna = dynamic_perturbation_factor

        # Running the Simulated Annealing heuristic
        for _ in tqdm(range(max_iter), disable=(not self.verbose)):
            
            # Generating a neighbor sample
            neighbor_sample, queries = neighborhood.generate_valid(
                current_sample,
                estimator=self.estimator,
                y=y_initial,
                is_targeted_attack=targeted,
                targeted_class=specific_class,
                static_perturbation_factor=static_perturbation_factor,
                dynamic_perturbation_factor=dyna,
                enable_negative_inflation_values=enable_negative_inflation_vector,
                inflation_vector_max_perturbation=inflation_vector_max_perturbation,
            )
            
            #print("is predict(neighbor_sample) == y_init: ", self.estimator.predict([neighbor_sample])[0] == y_initial)
            #print("is predict(current_sample) == y_init: ", self.estimator.predict([current_sample])[0] == y_initial)
            total_queries += queries

            # If neighbor_sample is a list, select one sample randomly
            if isinstance(neighbor_sample, list):
                neighbor_sample = random.choice(neighbor_sample)

            # Computing the cost difference
            neighbor_cost = cost_function(neighbor_sample, input)
            delta_E = neighbor_cost - current_cost

            # Acceptance criterion
            if delta_E < 0:
                # Accept the new sample unconditionally
                current_sample = neighbor_sample
                current_cost = neighbor_cost
            else:
                # Compute acceptance probability
                acceptance_probability = np.exp(-delta_E / T)
                if np.random.rand() < acceptance_probability:
                    # Accept the new sample probabilistically
                    current_sample = neighbor_sample
                    current_cost = neighbor_cost

            # Updating the best sample found so far
            if current_cost < best_cost:
                p = patience
                dyna = dynamic_perturbation_factor
                best_sample = current_sample.copy()
                best_cost = current_cost
                #assert self.estimator.predict([best_sample])[0] != y_initial
                heuristic_history.append([best_sample, best_cost])
            else:
                p -= 1
                if p <= 0:
                    break
                if raise_dynamic_perturbation_after_non_improving_candidate:
                    dyna *= dynamic_perturbation_factor

            # Decreasing the temperature
            T *= cooling_rate

            # Optional early stopping if temperature is very low
            if T < final_temperature:
                break

        if not heuristic_history:
            raise ValueError("Impossible to find a sample satisfying constraints and misclassification.")
    
        
        # Select best sample associated with the lowest cost in heuristic history
        best_sample, best_cost = min(heuristic_history, key=lambda x: x[1])

        self.heuristic_history.append(heuristic_history)

        # Final check to ensure the best sample is adversarial
        total_queries += 1
        final_prediction = self.estimator.predict([best_sample])[0]
        if (targeted and final_prediction != specific_class) or (not targeted and final_prediction == y_initial):
            raise ValueError("This is embarrassing... The final sample is not adversarial!")

        return best_sample, best_cost, total_queries