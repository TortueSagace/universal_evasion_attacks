"""
This child of the "Attack" class implements the Particle Swarm Optimization algorithm.
It uses a swarm of particles to explore the search space and generate adversarial examples.
"""

from universal_evasion_attacks.config import params
from universal_evasion_attacks import L2_norm, Radar, Flower, Lightning, random, tqdm, np, Attack

# Mapping from strings to actual functions/classes
object_map = {
    "L2_norm": L2_norm,
    "Radar": Radar,
    "Flower": Flower,
    "Lightning": Lightning
}

# Extracting the default parameters for ParticleSwarm attack
params_attack = params["attack"]["ParticleSwarm"]

# Defining the ParticleSwarm class
class ParticleSwarm(Attack):
    """
    Child of the Attack class that implements the Particle Swarm Optimization algorithm.
    """
    def __init__(self, estimator=None, verbose:int=1):
        """
        Constructor for the ParticleSwarm class.
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
        Method for executing the Particle Swarm Optimization algorithm.

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
        swarm_size = params['swarm_size']
        inertia_weight = params['inertia_weight']
        cognitive_coefficient = params['cognitive_coefficient']
        social_coefficient = params['social_coefficient']
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

        # Initialize the swarm
        perturbation_weights = [static_perturbation_factor] * len(input)
        total_queries = 0

        # Generate initial swarm positions
        swarm_positions, queries = neighborhood.generate_valid(
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
            num_samples=swarm_size
        )
        if not isinstance(swarm_positions, list):
            raise ValueError(f"The neighborhood's generate_valid method must return a list of samples for ParticleSwarm, got {type(swarm_positions)}.")
        total_queries += queries

        self.verbose_message("Initial swarm positions generated.")

        # Initialize velocities
        swarm_velocities = [np.zeros_like(input) for _ in range(swarm_size)]

        # Evaluate fitness of the swarm
        fitness_scores = [cost_function(particle, input) for particle in swarm_positions]

        # Initialize personal best positions and scores
        personal_best_positions = list(swarm_positions)
        personal_best_scores = list(fitness_scores)

        # Initialize global best position and score
        global_best_index = np.argmin(personal_best_scores)
        global_best_position = personal_best_positions[global_best_index]
        global_best_score = personal_best_scores[global_best_index]

        # Initialize history
        self.heuristic_history.append([])

        # Particle Swarm Optimization main loop
        for iteration in tqdm(range(max_iterations), disable=(not self.verbose)):
            for i in range(swarm_size):
                # Update velocity
                inertia = inertia_weight * swarm_velocities[i]
                cognitive_component = cognitive_coefficient * np.random.rand(len(input)) * (personal_best_positions[i] - swarm_positions[i])
                social_component = social_coefficient * np.random.rand(len(input)) * (global_best_position - swarm_positions[i])
                swarm_velocities[i] = inertia + cognitive_component + social_component

                # Update position
                swarm_positions[i] = swarm_positions[i] + swarm_velocities[i]

                # Ensure new position satisfies constraints
                adjusted_position, q = neighborhood.generate_valid(
                    swarm_positions[i],
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
                swarm_positions[i] = adjusted_position[0]

                # Evaluate new fitness
                fitness_scores[i] = cost_function(swarm_positions[i], input)
                total_queries += 1

                # Update personal best
                if fitness_scores[i] < personal_best_scores[i]:
                    personal_best_positions[i] = swarm_positions[i]
                    personal_best_scores[i] = fitness_scores[i]

                    # Update global best
                    if personal_best_scores[i] < global_best_score:
                        global_best_position = personal_best_positions[i]
                        global_best_score = personal_best_scores[i]
                        self.verbose_message(f"Iteration {iteration}: New global best score {global_best_score}.")

                # Record history
                self.heuristic_history[-1].append([swarm_positions[i], fitness_scores[i]])

                # Check if current particle is adversarial
                prediction = self.estimator.predict([swarm_positions[i]])[0]
                if (targeted and prediction == specific_class) or (not targeted and prediction != y_initial):
                    self.verbose_message(f"Adversarial example found at iteration {iteration}.")
                    return swarm_positions[i], fitness_scores[i], total_queries

        if global_best_position is None or global_best_score == np.inf:
            raise ValueError("Impossible to find a sample satisfying constraints and misclassification.")

        # Final check to ensure the global best position is adversarial
        total_queries += 1
        final_prediction = self.estimator.predict([global_best_position])[0]
        if (targeted and final_prediction != specific_class) or (not targeted and final_prediction == y_initial):
            raise ValueError("This is embarrassing... The final sample is not adversarial!")

        return global_best_position, global_best_score, total_queries