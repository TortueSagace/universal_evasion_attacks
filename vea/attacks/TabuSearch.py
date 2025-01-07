"""
This child of the "Attack" class implements the Tabu Search heuristic.
It uses a tabu list to keep track of recently visited solutions to avoid cycles and escape local minima.
"""

from vea.config import params
from vea import L2_norm, Radar, Flower, Lightning, random, tqdm, np, Attack

# Mapping from strings to actual functions/classes
object_map = {
    "L2_norm": L2_norm,
    "Radar": Radar,
    "Flower": Flower,
    "Lightning": Lightning,
}

# Extracting the default parameters for TabuSearch attack
params_attack = params["attack"]["TabuSearch"]

# Defining the TabuSearch class
class TabuSearch(Attack):
    """
    Child of the Attack class that implements the Tabu Search heuristic.
    """
    def __init__(self, estimator=None, verbose:int=1):
        """
        Constructor for the TabuSearch class.
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
        Method for executing the Tabu Search heuristic.

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
        inflation_vector_max_perturbation = params['inflation_vector_max_perturbation']
        enable_negative_inflation_vector = params['enable_negative_inflation_vector']
        patience = params['patience']
        raise_dynamic_perturbation_after_non_improving_candidate = params['raise_dynamic_perturbation_after_non_improving_candidate']
        tabu_tenure = params['tabu_tenure']
        max_tabu_size = params['max_tabu_size']
        similarity_epsilon = params['similarity_epsilon']

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
        current_sample = input.copy()
        best_sample = None
        current_cost = np.inf
        best_cost = current_cost
        total_queries = 0

        # Initialize the tabu list (list of moves or solutions)
        tabu_list = []

        # Initialize the patience
        p = patience

        # Initialize dynamic factor
        dyna = dynamic_perturbation_factor

        # Initialize history (single agent)
        heuristic_history = []

        # Running the Tabu Search heuristic
        for iteration in tqdm(range(max_iter), disable=(not self.verbose)):
            # Generating neighbor samples
            neighbor_samples, queries = neighborhood.generate_valid(
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
            total_queries += queries

            # If neighbor_samples is not a list, make it a list
            if not isinstance(neighbor_samples, list):
                neighbor_samples = [neighbor_samples]

            # Evaluate neighbors and filter out tabu moves and non-adversarial samples
            candidate_list = []
            for neighbor in neighbor_samples:
                move = neighbor - current_sample  # Define the move
                # Check if move is in tabu list
                is_tabu = any(np.linalg.norm(move - tabu_move['move'], ord=1) < similarity_epsilon for tabu_move in tabu_list)
                neighbor_cost = cost_function(neighbor, input)

                if is_tabu:
                    # Aspiration criterion: if the move is tabu but improves the best cost, consider it
                    if neighbor_cost < best_cost:
                        candidate_list.append((neighbor, neighbor_cost, move))
                else:
                    candidate_list.append((neighbor, neighbor_cost, move))

            # If no candidates are available (all moves are tabu or non-adversarial), handle accordingly
            if not candidate_list:
                p -= 1
                if p <= 0:
                    break
                if raise_dynamic_perturbation_after_non_improving_candidate:
                    dyna *= dynamic_perturbation_factor
                continue

            # Select the best candidate
            candidate_list.sort(key=lambda x: x[1])  # Sort by cost
            best_candidate = candidate_list[0]
            next_sample, next_cost, move = best_candidate

            # Update current sample and cost
            current_sample = next_sample.copy()
            current_cost = next_cost

            # Update tabu list
            tabu_list.append({'move': move, 'tenure': tabu_tenure})
            if len(tabu_list) > max_tabu_size:
                tabu_list.pop(0)

            # Decrease tenure of moves in tabu list and remove expired ones
            for tabu_move in tabu_list[:]:
                tabu_move['tenure'] -= 1
                if tabu_move['tenure'] <= 0:
                    tabu_list.remove(tabu_move)

            # Updating the best sample found so far
            if current_cost < best_cost:
                best_sample = current_sample.copy()
                best_cost = current_cost
                # Update history
                heuristic_history.append([current_sample.copy(), current_cost])
                p = patience
                dyna = dynamic_perturbation_factor
            else:
                p -= 1
                if p <= 0:
                    break
                if raise_dynamic_perturbation_after_non_improving_candidate:
                    dyna *= dynamic_perturbation_factor

        if best_sample is None or best_cost == np.inf:
            raise ValueError("Impossible to find a sample satisfying constraints and misclassification.")
        
        # Select best sample associated with the lowest cost in heuristic history
        best_sample, best_cost = min(heuristic_history, key=lambda x: x[1])

        # Final check to ensure the best sample is adversarial
        total_queries += 1
        final_prediction = self.estimator.predict([best_sample])[0]
        if (targeted and final_prediction != specific_class) or (not targeted and final_prediction == y_initial):
            raise ValueError("This is embarrassing... The final sample is not adversarial!")

        # Append the heuristic history to the class's history
        self.heuristic_history.append(heuristic_history)

        return best_sample, best_cost, total_queries