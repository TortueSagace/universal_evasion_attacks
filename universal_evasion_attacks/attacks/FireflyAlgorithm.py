"""
This child of the "Attack" class implements the Firefly Algorithm.
It uses the attraction mechanism among fireflies to generate adversarial examples.
"""

from universal_evasion_attacks.config import params
from universal_evasion_attacks import L2_norm, Radar, Flower, Lightning, random, tqdm, np, Attack

# Mapping from strings to actual functions/classes
object_map = {
    "L2_norm": L2_norm,
    "Radar": Radar,
    "Flower": Flower,
    "Lightning": Lightning,
}

# Extracting the default parameters for FireflyAlgorithm attack
params_attack = params["attack"]["FireflyAlgorithm"]

# Defining the FireflyAlgorithm class
class FireflyAlgorithm(Attack):
    """
    Child of the Attack class that implements the Firefly Algorithm.
    """
    def __init__(self, estimator=None, verbose:int=1):
        """
        Constructor for the FireflyAlgorithm class.
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
        Method for executing the Firefly Algorithm.

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
        population_size = params['population_size']
        max_generations = params['max_generations']
        absorption_coefficient = params['absorption_coefficient']  # γ
        attractiveness_at_zero = params['attractiveness_at_zero']  # β0
        step_size = params['step_size']  # α
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

        # Initialize the population (fireflies)
        perturbation_weights = [static_perturbation_factor] * len(input)
        total_queries = 0

        # Generate initial population
        population, queries = neighborhood.generate_valid(
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
            num_samples=population_size
        )
        if not isinstance(population, list):
            raise ValueError(f"The neighborhood's generate_valid method must return a list of samples for FireflyAlgorithm, got {type(population)}.")
        total_queries += queries

        self.verbose_message("Initial population generated.")

        # Evaluate fitness of the population
        fitness_scores = [cost_function(individual, input) for individual in population]

        # Initialize history
        self.heuristic_history.append([])

        best_sample = None
        best_cost = np.inf

        # Firefly Algorithm main loop
        for generation in tqdm(range(max_generations), disable=(not self.verbose)):
            for i in range(population_size):
                for j in range(population_size):
                    if fitness_scores[j] < fitness_scores[i]:
                        # Calculate distance between fireflies
                        rij = np.linalg.norm(population[i] - population[j])
                        # Calculate attractiveness
                        beta = attractiveness_at_zero * np.exp(-absorption_coefficient * rij ** 2)
                        # Move firefly i towards firefly j
                        population[i] = population[i] + beta * (population[j] - population[i]) + \
                                        step_size * (np.random.rand(len(input)) - 0.5)

                        # Ensure new position satisfies constraints
                        adjusted_position, q = neighborhood.generate_valid(
                            population[i],
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
                        population[i] = adjusted_position[0]

                        # Evaluate new fitness
                        fitness_scores[i] = cost_function(population[i], input)
                        total_queries += 1

                        # Check if new position is adversarial
                        prediction = self.estimator.predict([population[i]])[0]
                        if (targeted and prediction == specific_class) or (not targeted and prediction != y_initial):
                            self.verbose_message(f"Adversarial example found at generation {generation}.")
                            return population[i], fitness_scores[i], total_queries

            # Update best sample
            min_index = np.argmin(fitness_scores)
            if fitness_scores[min_index] < best_cost:
                best_sample = population[min_index]
                best_cost = fitness_scores[min_index]
                self.verbose_message(f"Generation {generation}: New best cost {best_cost}.")

            # Record history
            for individual, fitness in zip(population, fitness_scores):
                self.heuristic_history[-1].append([individual, fitness])

        if best_sample is None or best_cost == np.inf:
            raise ValueError("Impossible to find a sample satisfying constraints and misclassification.")

        # Final check to ensure the best sample is adversarial
        total_queries += 1
        final_prediction = self.estimator.predict([best_sample])[0]
        if (targeted and final_prediction != specific_class) or (not targeted and final_prediction == y_initial):
            raise ValueError("This is embarrassing... The final sample is not adversarial!")

        return best_sample, best_cost, total_queries