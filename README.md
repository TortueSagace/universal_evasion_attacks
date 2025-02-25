# Versatile Evasion Attacks

This repository contains the final code associated to my master's thesis "Metaheuristic-based Evasion Attacks on State-of-the-art Near-perfect Machine Learning Classifiers", supervised by Pr. Jean-Michel Dricot,President of jury of the Master of Science in Cybersecurity at the Université Libre de Bruxelles (ULB).

## Context

An evasion attack is a process that aims to find a perturbation of the input data that is small enough to be imperceptible to a human, but that is sufficient to change the prediction of a machine learning model. It has been proven in several articles that deep neural networks for image classification are particularly sensitive to such attacks, leading to very dangerous situations in the context of security (e.g., driverless cars) and privacy (e.g., face recognition).

![Famous example](https://miro.medium.com/v2/resize:fit:828/format:webp/1*zlxO7NuxK6ZijZtXnRl46Q.png)

*Image source: [Explaining and Harnessing Adversarial Examples](https://arxiv.org/abs/1412.6572v3)*

Evasion attacks on image classifiers were widely used, but tabular data classifiers are also sensitive to such attacks, and the literature is considerably less developed in this area. The goal of this thesis was to develop a set of versatile evasion attacks that can be applied on both image and tabular data classifiers (hence the term "versatile"), with a particular focus on the latter. A series of different constraints types can be considered by the algorithms (equalities, inequalities, ...), allowing the evasion attacks to be implemented in realistic black-box and white-box scenarios (in particular, decision-based and transfer-based attacks).

## Content

This package contains security protocols for estimating adversarial robustness of machine learning models for both tabular and image datasets. It implements a set of evasion attacks based on heuristic optimization algorithms, and complex cost functions to give reliable results for tabular problems. Three protocols can be distinguished:
1. **Image protocol**: this protocol is designed to evaluate the robustness of image classifiers. It uses the L2 norm as a cost function.
2. **Profit protocol**: this protocol is designed to evaluate the robustness of tabular data classifiers, where the success of the attack can be estimated through a gain-cost approach. This implies that the features of the dataset are interpretable enough so that a custom cost function can be built in a reasonable time. This does not apply to "obscure" tabular datasets, where it is almost impossible to build such profit functions.
3. **HAR protocol**: for the latter case, the HAR protocol is used. "HAR" stands for Heuristic Adversarial Robustness. This metric goes from 0 to 1, with 1 being a "perfect" robustness, and 0.5 corresponding to a mediocre robustness. It uses several complex statistical metrics to evaluate the success of the attack (including feature importance towards the target class), even when the features are not interpretable. This protocol can only be used in targeted evasion attacks. It comes with a series of checks that ensure the reliability of the results before applying the algorithms. All details about cost functions and metrics can be found in the thesis at chapter 5.


The two first protocols are merged in a single function "master.protocols.custom_cost_protocol", in which the user can choose the protocol to use. The HAR protocol is implemented in the "master.protocols.HAR_protocol" function.


The following subfolders are present in the repository:

- **attacks**: contains heuristic attacks algorithms, wrapped into classes. HillClimbing, SimulatedAnnealing and TabuSearch are fully implemented, the other attacks are still in development.
- **master**: contains the "Master" class, which allows to automate several processes and change the global parameters of the attacks, neighborhoods and default machine learning classifiers. It also contains the "protocols.py" script, in which "HAR_protocol" and "custom_cost_protocol" are defined.
- **neighborhoods**: contains neighborhood classes associated with heuristic attacks. The choice of the neighborhood is independent of the choice of the attack. Only the "Balloon" (previously named "Radar" in release 1.1.4) neighborhood is fully implemented for the moment, but it is versatile enough to be used with any attack.

This library was successfully tested on several datasets with different attack and machine learning classifiers (including image classifiers and tabular data classifiers). The link to those dataset can be found in the [useful_links](./useful_links/) directory of this repository. Please note that attacks on deep neural networks were not tested yet, but it should work as well as the other classifiers tested in my work.

## Dependencies

Please note that there are several dependencies to run the code. The following libraries are required (we do not force the version of the libraries, but the code was tested with the following versions):


- Python version: 3.13.0 (tags/v3.13.0:60403a5, Oct  7 2024, 09:38:07) [MSC v.1941 64 bit (AMD64)]
- pandas: 2.2.3
- numpy: 2.1.3
- seaborn: 0.13.2
- tqdm: 4.67.0
- xgboost: 2.1.2
- lightgbm: 4.5.0
- sklearn: 1.5.2
- optuna: 4.0.0
- joblib: 1.4.2
- seaborn: 0.13.2
- matplotlib: 3.9.2

## Simple example

Here is a simple example of how to use the library. The complete juptyer notebook I used to generate results for chapter 6 of my master's thesis is available in the "examples" folder. Please not however that the data dependencies are not included in the repository.

```python
from vea import HAR_protocol

# Dataset contained in a pandas format. All categorical features must be numerically encoded.
X, y = my_dataframe.drop(columns=["target"]), my_dataframe["target"]

# Define the constraints for your specific evasion problem
my_constraints = {
    "equality": ["x[4] = 0.0", "x[5] = 0.0", "x[6] = 1.0"], # cannot change the values of these features
    "inequality": ["x[2] >= x[1]"], # (inequality constaints were fully implemented but not tested yet)
    "clip_min": list(X.quantile(0.01)), 
    "clip_max": list(X.quantile(0.99)),
    "categorical": [None]*(X.shape[1]) # this means you don't have any categorical features
}

# Apply the HAR protocol
results = HAR_protocol(X=X, y=y, # The only mandatory arguments
                target_class=0, # The class you want to "look like"
                constraints=my_constraints, # Optional. Default will only take 0.01 and 0.99 clipping constraints
                verbose=2, # Optional. 0: no print, 1: print main steps, 2: print everything
                ) # If no model is provided, a default model is used (LightGBM)

# Save the results in a pandas dataframe
results_df = pd.DataFrame([results], columns=["HAR", "confidence_score", "adversarial_accuracy", "avg_likelihood", "max_likelihood"])
```

The evasion attacks contain multiple parameters. The default parameter configurations can be found in "master.master_params.json". They can be changed by the Master class:

```python
from vea import Master

master = Master(verbosity=1) # execute once

# Step 1: Inspection
for key, value in master.params["attack"]["HillClimbing"].items():
    print(key, ":", value)

# Step 1: Modification
master.update_params("attack", "HillClimbing", {"dynamic_perturbation_factor": 1.05})
print(f"\nNew dynamic perturbation factor: {master.params["attack"]["HillClimbing"]["dynamic_perturbation_factor"]}")

# Step 3: Rollback (at the end of the experiment)
master.rollback_json(category="attack", name="HillClimbing")
```

## Compatibility with ART

This library is compatible with the [Adversarial Robustness Toolbox (ART) library](https://github.com/Trusted-AI/adversarial-robustness-toolbox). To include specific ART attacks in the protocols, you simply need to wrap those attacks inside an Attack child class containing an appropriate "run" method (see the "attacks" folder). This might involve some work, but it is feasible in a reasonable time. After that, you can use the attack in the protocols as any other attack:

```python
from vea import custom_cost_protocol

results = custom_cost_protocol(X=X, y=y,
                targeted=False,
                images=True,
                image_shape=my_image_shape,
                constraints=my_constraints, # Will not be considered by your ART attack unless you modify it
                attacks = [MyCustomARTAttack(), HillClimbing(), SimulatedAnnealing()],
                verbose=2)
```                

## Contributions

This library is still in development. If you want to contribute, please feel free to fork the repository and submit a pull request. I will be happy to review it. The main areas of improvement are the following:
- Finish the implementation of the neighborhoods Flower and Lightning in the "neighborhoods" folder according the their respective descriptions in chapter 4 of my thesis.
- Finish the implementation of the attacks in the "attacks" folder.
- Test the inequality constraints (those are implemented in the neighborhoods mother class)
- Add scaling possibilities for neighborhood search (currently, the neighborhood algorithm makes equal steps in all directions, but taking into account the standard deviation of the features could be interesting)
- The `trivial_success_if_already_target` is currently available in `custom_cost_protocol` but not yet in `HAR_protocol`.
- Improve ART compatibility
- Test with deep neural networks
- Compare evasion attacks performances with other tabular evasion attacks like LowProFool
- Implement RegSLARC, that also penalizes weighted distance between the original and adversarial examples
- Improve the HAR protocol if necessary, by using the RegSLARC metric for instance
- Implement other protocols for obscure tabular datasets if necessary
- Optimizing the code for better performance if possible

## License

This code is distributed under the MIT license. Feel free to use it for your own research, but please cite the thesis "Metaheuristic-based Evasion Attacks on State-of-the-art Near-perfect Machine Learning Classifiers" by Alexandre Le Mercier (will be published later in 2025) if you use it in a publication.


Happy coding!
