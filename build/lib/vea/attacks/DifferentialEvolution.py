"""
This child of the "Attack" class implements the Differential Evolution algorithm.
It uses a population-based evolutionary approach to generate adversarial examples.
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

# Extracting the default parameters for DifferentialEvolution attack
params_attack = params["attack"]["DifferentialEvolution"]

# Defining the DifferentialEvolution class
class DifferentialEvolution(Attack):
    """
    Child of the Attack class that implements the Differential Evolution algorithm.
    """
    def __init__(self, estimator=None, verbose:int=1):
        """
        Constructor for the DifferentialEvolution class.
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
        Method for executing the Differential Evolution algorithm.

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
        mutation_factor = params['mutation_factor']  # F in DE
        crossover_probability = params['crossover_probability']  # CR in DE
        max_generations = params['max_generations']
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

        # Initialize the population
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
            raise ValueError(f"The neighborhood's generate_valid method must return a list of samples for DifferentialEvolution, got {type(population)}.")
        total_queries += queries

        self.verbose_message("Initial population generated.")

        # Evaluate fitness of the population
        fitness_scores = [cost_function(individual, input) for individual in population]

        # Initialize history
        self.heuristic_history.append([])

        best_sample = None
        best_cost = np.inf

        # Differential Evolution main loop
        for generation in tqdm(range(max_generations), disable=(not self.verbose)):
            new_population = []
            for i in range(population_size):
                target_vector = population[i]
                indices = list(range(population_size))
                indices.remove(i)
                a, b, c = random.sample(indices, 3)
                x1 = population[a]
                x2 = population[b]
                x3 = population[c]

                # Mutation
                mutant_vector = x1 + mutation_factor * (x2 - x3)

                # Crossover
                trial_vector = np.copy(target_vector)
                for j in range(len(input)):
                    if np.random.rand() < crossover_probability:
                        trial_vector[j] = mutant_vector[j]

                # Ensure trial_vector satisfies constraints
                # Adjust the trial_vector using the neighborhood's generate method
                adjusted_vector, q = neighborhood.generate_valid(
                    trial_vector,
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
                trial_vector = adjusted_vector[0]

                # Evaluate trial vector
                trial_fitness = cost_function(trial_vector, input)
                total_queries += 1

                # Selection
                if trial_fitness < fitness_scores[i]:
                    new_population.append(trial_vector)
                    fitness_scores[i] = trial_fitness
                else:
                    new_population.append(target_vector)

                # Update history and best sample
                self.heuristic_history[-1].append([new_population[-1], fitness_scores[i]])
                if fitness_scores[i] < best_cost:
                    best_sample = new_population[-1]
                    best_cost = fitness_scores[i]

                # Check if trial_vector is adversarial
                prediction = self.estimator.predict([new_population[-1]])[0]
                if (targeted and prediction == specific_class) or (not targeted and prediction != y_initial):
                    self.verbose_message(f"Adversarial example found in generation {generation}.")
                    return new_population[-1], fitness_scores[i], total_queries

            population = new_population

        if best_sample is None or best_cost == np.inf:
            raise ValueError("Impossible to find a sample satisfying constraints and misclassification.")

        # Final check to ensure the best sample is adversarial
        total_queries += 1
        final_prediction = self.estimator.predict([best_sample])[0]
        if (targeted and final_prediction != specific_class) or (not targeted and final_prediction == y_initial):
            raise ValueError("This is embarrassing... The final sample is not adversarial!")

        return best_sample, best_cost, total_queries