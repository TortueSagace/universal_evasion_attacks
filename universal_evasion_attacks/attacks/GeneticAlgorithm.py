"""
This child of the "Attack" class implements the Genetic Algorithm heuristic.
It uses a population-based evolutionary approach to generate adversarial examples.
"""

import sys
import json
import numpy as np
import random
from tqdm import tqdm
sys.path.append("../master")

# Importing necessary libraries
from attacks.header import *
from attacks.Attack import Attack
from neighborhoods.Neighborhood import Neighborhood
from neighborhoods.Radar import Radar
from neighborhoods.Flower import Flower
from neighborhoods.Lightning import Lightning

# Load parameters from JSON
with open('../master/master_params.json', 'r') as file:
    params = json.load(file)

# Mapping from strings to actual functions/classes
object_map = {
    "L2_norm": L2_norm,
    "Radar": Radar,
    "Flower": Flower,
    "Lightning": Lightning
}

# Extracting the default parameters for GeneticAlgorithm attack
params_attack = params["attack"]["GeneticAlgorithm"]

# Defining the GeneticAlgorithm class
class GeneticAlgorithm(Attack):
    """
    Child of the Attack class that implements the Genetic Algorithm heuristic.
    """
    def __init__(self, estimator=None, verbose:int=1):
        """
        Constructor for the GeneticAlgorithm class.
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
        Method for executing the Genetic Algorithm heuristic.

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
        crossover_rate = params['crossover_rate']
        mutation_rate = params['mutation_rate']
        max_generations = params['max_generations']
        static_perturbation_factor = params['static_perturbation_factor']
        dynamic_perturbation_factor = params['dynamic_perturbation_factor']
        inflation_vector_max_perturbation = params['inflation_vector_max_perturbation']
        enable_negative_inflation_vector = params['enable_negative_inflation_vector']
        selection_method = params['selection_method']
        tournament_size = params.get('tournament_size', 3)  # Default value if not specified
        elitism = params.get('elitism', True)

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
        population = []
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
            raise ValueError(f"The neighborhood's generate_valid method must return a list of samples for GeneticAlgorithm, got {type(population)}: {population}.")
        total_queries += queries

        self.verbose_message("Initial population generated.")

        # Evaluate fitness of the population
        fitness_scores = [cost_function(individual, input) for individual in population]

        # Initialize history
        self.heuristic_history.append([])

        best_sample = None
        best_cost = np.inf

        # Genetic Algorithm main loop
        for generation in tqdm(range(max_generations), disable=(not self.verbose)):
            new_population = []

            # Elitism: Carry over the best individual to the next generation
            if elitism:
                elite_index = np.argmin(fitness_scores)
                elite_individual = population[elite_index]
                new_population.append(elite_individual)
                self.verbose_message(f"Generation {generation}: Elite individual carried over.")

            while len(new_population) < population_size:
                # Selection
                parent1 = self.selection(population, fitness_scores, method=selection_method, tournament_size=tournament_size)
                parent2 = self.selection(population, fitness_scores, method=selection_method, tournament_size=tournament_size)

                # Crossover
                if np.random.rand() < crossover_rate:
                    offspring1, offspring2 = self.crossover(parent1, parent2)
                else:
                    offspring1, offspring2 = parent1.copy(), parent2.copy()

                # Mutation
                offspring1, q = self.mutate(offspring1, mutation_rate, neighborhood, perturbation_weights,
                                            y_initial, targeted, specific_class, static_perturbation_factor,
                                            dynamic_perturbation_factor, inflation_vector_max_perturbation,
                                            enable_negative_inflation_vector)
                queries += q
                offspring2, q = self.mutate(offspring2, mutation_rate, neighborhood, perturbation_weights,
                                            y_initial, targeted, specific_class, static_perturbation_factor,
                                            dynamic_perturbation_factor, inflation_vector_max_perturbation,
                                            enable_negative_inflation_vector)
                queries += q

                new_population.extend([offspring1, offspring2])

            # Ensure the population size is maintained
            population = new_population[:population_size]

            # Evaluate fitness of the new population
            fitness_scores = [cost_function(individual, input) for individual in population]

            # Update history and best sample
            for individual, fitness in zip(population, fitness_scores):
                self.heuristic_history[-1].append([individual, fitness])
                if fitness < best_cost:
                    best_sample = individual
                    best_cost = fitness

            # Check for adversarial examples in the population
            predictions = self.estimator.predict(population)
            for i, prediction in enumerate(predictions):
                if (targeted and prediction == specific_class) or (not targeted and prediction != y_initial):
                    self.verbose_message(f"Adversarial example found in generation {generation}.")
                    return population[i], fitness_scores[i], total_queries

            # Increment total queries
            total_queries += len(population)

        if best_sample is None or best_cost == np.inf:
            raise ValueError("Impossible to find a sample satisfying constraints and misclassification.")

        # Final check to ensure the best sample is adversarial
        total_queries += 1
        final_prediction = self.estimator.predict([best_sample])[0]
        if (targeted and final_prediction != specific_class) or (not targeted and final_prediction == y_initial):
            raise ValueError("This is embarrassing... The final sample is not adversarial!")

        return best_sample, best_cost, total_queries

    def selection(self, population, fitness_scores, method='tournament', tournament_size=3):
        """
        Selection method for choosing parents.

        Parameters:
            population: List of individuals.
            fitness_scores: Corresponding fitness scores.
            method: Selection method ('tournament' or 'roulette').
            tournament_size: Size of the tournament for tournament selection.

        Returns:
            Selected individual.
        """
        if method == 'tournament':
            # Tournament selection
            participants = random.sample(list(zip(population, fitness_scores)), tournament_size)
            participants.sort(key=lambda x: x[1])
            return participants[0][0]
        elif method == 'roulette':
            # Roulette wheel selection
            total_fitness = sum(fitness_scores)
            selection_probs = [fitness / total_fitness for fitness in fitness_scores]
            return population[np.random.choice(len(population), p=selection_probs)]
        else:
            raise ValueError("Unknown selection method.")

    def crossover(self, parent1, parent2):
        """
        Crossover operation between two parents.

        Parameters:
            parent1: First parent individual.
            parent2: Second parent individual.

        Returns:
            Two offspring individuals.
        """
        # Single-point crossover
        crossover_point = np.random.randint(1, len(parent1))
        offspring1 = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
        offspring2 = np.concatenate([parent2[:crossover_point], parent1[crossover_point:]])
        return offspring1, offspring2

    def mutate(self, individual, mutation_rate, neighborhood, perturbation_weights,
               y_initial=None, targeted=False, specific_class=None, static_perturbation_factor=0.1,
               dynamic_perturbation_factor=0.5, inflation_vector_max_perturbation=0.1,
               enable_negative_inflation_vector=False):
        """
        Mutation operation on an individual.

        Parameters:
            individual: The individual to mutate.
            mutation_rate: Mutation rate.
            neighborhood: Neighborhood instance.
            perturbation_weights: Perturbation weights.

        Returns:
            Mutated individual.
        """
        # Mutation with neighborhood's generate method
        qu = 0
        if np.random.rand() < mutation_rate:
            mutated_individuals, qu = neighborhood.generate_valid(
                individual,
                estimator=self.estimator,
                y=y_initial,
                is_targeted_attack=targeted,
                targeted_class=specific_class,
                static_perturbation_factor=static_perturbation_factor,
                dynamic_perturbation_factor=dynamic_perturbation_factor,
                inflation_vector_max_perturbation=inflation_vector_max_perturbation,
                enable_negative_inflation_values=enable_negative_inflation_vector,
                initial_perturbation_vector=perturbation_weights,
                num_samples=1 # we only need 1 sample
            )
            mutated_individual = mutated_individuals[0]
            return mutated_individual, qu
        else:
            return individual.copy(), qu