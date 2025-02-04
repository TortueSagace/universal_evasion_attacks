"""
This child of the "Attack" class implements the Harmony Search algorithm.
It uses a memory-based approach to generate adversarial examples.
"""

from vea.config import params
from vea import L2_norm, Balloon, Flower, Lightning, random, tqdm, np, Attack

# Mapping from strings to actual functions/classes
object_map = {
    "L2_norm": L2_norm,
    "Balloon": Balloon,
    "Flower": Flower,
    "Lightning": Lightning,
}

# Extracting the default parameters for HarmonySearch attack
params_attack = params["attack"]["HarmonySearch"]

# Defining the HarmonySearch class
class HarmonySearch(Attack):
    """
    Child of the Attack class that implements the Harmony Search algorithm.
    """
    def __init__(self, estimator=None, verbose:int=1):
        """
        Constructor for the HarmonySearch class.
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
        Method for executing the Harmony Search algorithm.

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
        harmony_memory_size = params['harmony_memory_size']
        harmony_memory_consideration_rate = params['harmony_memory_consideration_rate']
        pitch_adjustment_rate = params['pitch_adjustment_rate']
        max_iterations = params['max_iterations']
        static_perturbation_factor = params['static_perturbation_factor']
        dynamic_perturbation_factor = params['dynamic_perturbation_factor']
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

        # Initialize the harmony memory
        perturbation_weights = [static_perturbation_factor] * len(input)
        total_queries = 0

        # Generate initial harmony memory
        harmony_memory, queries = neighborhood.generate_valid(
            input,
            estimator=self.estimator,
            y=y_initial,
            is_targeted_attack=targeted,
            targeted_class=specific_class,
            static_perturbation_factor=static_perturbation_factor,
            dynamic_perturbation_factor=dynamic_perturbation_factor,
            inflation_vector_max_perturbation=inflation_vector_max_perturbation,
            enable_negative_inflation_values=enable_negative_inflation_vector,
            initial_perturbation_vector=perturbation_weights,
            num_samples=harmony_memory_size
        )
        if not isinstance(harmony_memory, list):
            raise ValueError(f"The neighborhood's generate_valid method must return a list of samples for HarmonySearch, got {type(harmony_memory)}.")
        total_queries += queries

        self.verbose_message("Initial harmony memory generated.")

        # Evaluate fitness of the harmony memory
        fitness_scores = [cost_function(harmony, input) for harmony in harmony_memory]

        # Initialize history
        self.heuristic_history.append([])

        best_sample = None
        best_cost = np.inf

        # Harmony Search main loop
        for iteration in tqdm(range(max_iterations), disable=(not self.verbose)):
            new_harmony = np.zeros_like(input)
            for i in range(len(input)):
                if np.random.rand() < harmony_memory_consideration_rate:
                    # Memory consideration
                    index = np.random.randint(0, harmony_memory_size)
                    new_harmony[i] = harmony_memory[index][i]

                    # Pitch adjustment
                    if np.random.rand() < pitch_adjustment_rate:
                        # Adjust the pitch (i.e., mutate the value)
                        adjustment = np.random.uniform(-1, 1) * perturbation_weights[i]
                        new_harmony[i] += adjustment
                else:
                    # Random selection
                    adjustment = np.random.uniform(-1, 1) * perturbation_weights[i]
                    new_harmony[i] = input[i] + adjustment

            # Ensure new harmony satisfies constraints
            adjusted_harmony, q = neighborhood.generate_valid(
                new_harmony,
                estimator=self.estimator,
                y=y_initial,
                is_targeted_attack=targeted,
                targeted_class=specific_class,
                static_perturbation_factor=static_perturbation_factor,
                dynamic_perturbation_factor=dynamic_perturbation_factor,
                inflation_vector_max_perturbation=inflation_vector_max_perturbation,
                enable_negative_inflation_values=enable_negative_inflation_vector,
                initial_perturbation_vector=perturbation_weights,
                num_samples=1
            )
            queries += q
            new_harmony = adjusted_harmony[0]

            # Evaluate new harmony
            new_fitness = cost_function(new_harmony, input)
            total_queries += 1

            # Update harmony memory if new harmony is better than worst harmony
            worst_index = np.argmax(fitness_scores)
            if new_fitness < fitness_scores[worst_index]:
                harmony_memory[worst_index] = new_harmony
                fitness_scores[worst_index] = new_fitness

            # Update best sample
            best_index = np.argmin(fitness_scores)
            if fitness_scores[best_index] < best_cost:
                best_sample = harmony_memory[best_index]
                best_cost = fitness_scores[best_index]
                self.verbose_message(f"Iteration {iteration}: New best cost {best_cost}.")

            # Record history
            self.heuristic_history[-1].append([new_harmony, new_fitness])

            # Check if new harmony is adversarial
            prediction = self.estimator.predict([new_harmony])[0]
            if (targeted and prediction == specific_class) or (not targeted and prediction != y_initial):
                self.verbose_message(f"Adversarial example found at iteration {iteration}.")
                return new_harmony, new_fitness, total_queries

        if best_sample is None or best_cost == np.inf:
            raise ValueError("Impossible to find a sample satisfying constraints and misclassification.")

        # Final check to ensure the best sample is adversarial
        total_queries += 1
        final_prediction = self.estimator.predict([best_sample])[0]
        if (targeted and final_prediction != specific_class) or (not targeted and final_prediction == y_initial):
            raise ValueError("This is embarrassing... The final sample is not adversarial!")

        return best_sample, best_cost, total_queries