"""
A child of the Neighborhood class that generates a neighborhood based on the Flower strategy.
The Flower strategy modifies only one parameter at a time, selecting it randomly.
The output of the "generate" method is a list containing the generated samples.
"""

from vea.config import params
from vea import Neighborhood, random, np

# Extracting the default parameters for Flower neighborhood
params_neighborhood = params["neighborhood"]["Flower"]

# Defining the Flower class
class Flower(Neighborhood):
    """
    Child of the Neighborhood class that generates a neighborhood based on the Flower strategy.
    """
    def __init__(self, constraints: dict, **kwargs):
        """
        Constructor for the Flower class.

        Parameters:
            constraints: dict, the constraints for the neighborhood.
            **kwargs: Optional parameters that override defaults in self.params.
        """
        # Load default parameters from the JSON file
        self.params = params_neighborhood.copy()
        # Update with any overrides provided via kwargs
        self.params.update(kwargs)

        self.epsilon = self.params.get('epsilon', 1e-9)
        self.verbose = self.params.get('verbose', 0)
        self.max_iter_generation = self.params.get('max_iter_generation', 1000)
        self.enable_warning_message = self.params.get('enable_warning_message', True)
        self.num_samples = self.params.get('num_samples', 10)
        self.min_adversarial_samples_factor = self.params.get('min_adversarial_samples_factor', 0.5)
        super().__init__(constraints)

    def generate(self, x, perturbation_weights: list, return_inflation_vector=False) -> list:
        """
        Method for generating a neighborhood based on the Flower strategy.
        Parameters:
            x: list or numpy array, the original input.
            perturbation_weights: list, the weights for the perturbation (list of the same size as x).
            return_inflation_vector: bool, whether to return the inflation vector.
        Returns:
            list, the generated samples contained in a list.
        """
        samples = []
        inflation_vectors = []
        for _ in range(self.num_samples):
            # Copy the input to avoid modifying the original
            x_sample = x.copy()
            inflation_vector = [0] * len(x_sample)

            # Randomly select one parameter to modify
            index_to_modify = random.randint(0, len(x_sample) - 1)

            # Generate a perturbation for that parameter
            perturbation = np.random.uniform(-1, 1) * perturbation_weights[index_to_modify]

            # Modify the selected parameter
            x_sample[index_to_modify] += perturbation

            # Extracting the constraints
            equalities, inequalities, categorical, clip_min, clip_max = self.extract_constraints()

            # Apply constraints to the modified parameter
            # 1. Equality constraints
            for equality in equalities:
                exec(equality)

            # 2. Categorical constraints
            if categorical[index_to_modify] is not None:
                x_sample[index_to_modify] = np.random.choice(categorical[index_to_modify])

            # 3. Clipping constraints
            mini = clip_min[index_to_modify]
            maxi = clip_max[index_to_modify]
            if mini is not None:
                if x_sample[index_to_modify] < mini:
                    inflation_vector[index_to_modify] -= mini - x_sample[index_to_modify]
                x_sample[index_to_modify] = max(mini, x_sample[index_to_modify])
            if maxi is not None:
                if x_sample[index_to_modify] > maxi:
                    inflation_vector[index_to_modify] -= x_sample[index_to_modify] - maxi
                x_sample[index_to_modify] = min(maxi, x_sample[index_to_modify])

            # 4. Inequality constraints
            for inequality in inequalities:
                # Split the inequality into left-hand and right-hand expressions
                left, operator, right = inequality.split(" ")
                # Evaluate the expressions with the current values in x_sample
                left_value = eval(left)
                right_value = eval(right)

                # Adjust the values based on the operator
                if operator == "<" and not (left_value < right_value):
                    inflation_vector[index_to_modify] += right_value - left_value
                    exec(f"{left} = {right} - self.epsilon")
                elif operator == ">" and not (left_value > right_value):
                    inflation_vector[index_to_modify] -= left_value - right_value
                    exec(f"{left} = {right} + self.epsilon")
                elif operator == "<=" and not (left_value <= right_value):
                    inflation_vector[index_to_modify] += right_value - left_value
                    exec(f"{left} = {right}")
                elif operator == ">=" and not (left_value >= right_value):
                    inflation_vector[index_to_modify] -= left_value - right_value
                    exec(f"{left} = {right}")

            samples.append(x_sample)
            inflation_vectors.append(inflation_vector)

        if return_inflation_vector:
            return samples, inflation_vectors
        else:
            return samples

    def generate_valid(self, x, estimator, y=None, **kwargs):
        """
        Method for generating valid samples that satisfy the constraints of the evasion attack.

        Parameters:
            x: list or numpy array, the original input.
            estimator: Estimator, the estimator to use.
            y: The true label of the input sample.
            **kwargs: Optional parameters that override defaults in self.params.

        Returns:
            The generated valid sample and the number of queries made.
        """
        # Update parameters with any overrides provided via kwargs
        params = self.params.copy()
        params.update(kwargs)

        # Extract parameters
        is_targeted_attack = params.get('is_targeted_attack', False)
        targeted_class = params.get('specific_class', None)
        static_perturbation_factor = params.get('static_perturbation_factor', 1e-6)
        dynamic_perturbation_factor = params.get('dynamic_perturbation_factor', 1.2)
        inflation_vector_max_perturbation = params.get('inflation_vector_max_perturbation', 2)
        enable_negative_inflation_values = params.get('enable_negative_inflation_values', False)
        initial_perturbation_vector = params.get('initial_perturbation_vector', None)
        num_samples = params.get('num_samples', self.num_samples)  # Use overridden or default value

        default_num_samples = num_samples

        max_trials = self.max_iter_generation  # Maximum number of trials before giving up
        queries = 0  # Keep track of the number of queries used

        if y is None:
            y = estimator.predict([x])[0]

        if initial_perturbation_vector is not None:
            perturbation_weights = initial_perturbation_vector
        else:
            perturbation_weights = [static_perturbation_factor] * len(x)

        success = False
        adv_samples = []

        for k in range(max_trials):
            if self.enable_warning_message:
                if k == max_trials // 10:
                    print(f"Flower Warning: The number of trials exceeded 10% of maximum generation trials. Found {len(adv_samples)} valid samples.")
                elif k == max_trials // 4:
                    print(f"Flower Warning: The number of trials exceeded 25% of maximum generation trials. Found {len(adv_samples)} valid samples.")
                elif k == max_trials // 2:
                    print(f"Flower Warning: The number of trials exceeded 50% of maximum generation trials. Found {len(adv_samples)} valid samples.")

            queries += num_samples
            samples, inflation_vectors = self.generate(x, perturbation_weights, return_inflation_vector=True)

            if not enable_negative_inflation_values:
                # Put every negative inflation value to 0
                inflation_vectors = [[max(0, value) for value in vector] for vector in inflation_vectors]

            predictions = estimator.predict(samples)

            for i, sample in enumerate(samples):
                if is_targeted_attack:
                    if predictions[i] == targeted_class:
                        adv_samples.append(sample)
                else:
                    if predictions[i] != y:
                        adv_samples.append(sample)

            self.num_samples = default_num_samples - len(adv_samples)

            if len(adv_samples) == num_samples:
                success = True
                self.num_samples = default_num_samples
                break

            # If we reach this point, we have not found a valid sample for this iteration
            # Default behavior is to increase the perturbation weights
            # We compute the sigmoid function for each feature. At "0", this function returns "1"
            # Aggregate inflation vectors
            avg_inflation_vector = np.mean(inflation_vectors, axis=0)
            sigmoid = lambda x: inflation_vector_max_perturbation / (1 + np.exp(-x))

            # Update the perturbation weights with dynamic perturbation factor and the sigmoid function
            perturbation_weights = [perturbation_weights[i] * dynamic_perturbation_factor *
                                    sigmoid(avg_inflation_vector[i]) for i in range(len(x))]

        if success:
            return adv_samples, queries
        else:
            if np.inf in perturbation_weights:
                raise Exception(f"Flower: Overflow in the perturbation weights. Possible solutions:\n"
                                f"    1. Disable dynamic perturbation (set dynamic perturbation factor to 1.0)\n"
                                f"    2. Enable negative values in the inflation vector, and set min and max clips for every feature.")
            raise Exception(f"Flower: No valid sample satisfied the constraints after {max_trials} trials.")
