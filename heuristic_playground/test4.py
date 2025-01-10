import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from attacks.header import *
from neighborhoods.header import *
from attacks.HillClimbing import HillClimbing
from attacks.SimulatedAnnealing import SimulatedAnnealing
from attacks.TabuSearch import TabuSearch
from neighborhoods.Radar import Radar

# Define the maze as a 2D numpy array (0: wall, 1: free space)
maze = np.array([
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
])

maze = maze[::-1, :]
maze_size = len(maze)

class Model4:
    def predict(self, x):
        # Convert coordinates to maze indices with bounds checking
        col = int(np.floor(x[0]*maze_size))
        row = int(np.floor(x[1]*maze_size))

        row = np.clip(row, a_min=0, a_max=maze_size-1)
        col = np.clip(col, a_min=0, a_max=maze_size-1)

        # Return the maze value at this location (0 for wall, 1 for free space)
        return maze[row, col]

epsilon = 1e-2
def Cost4(x, x_initial):
    
    if 0.25 - epsilon < x[0] < 0.25 + epsilon and 0.25 - epsilon < x[1] < 0.25 + epsilon:
        return 3
    elif 0.25 - epsilon < x[0] < 0.25 + epsilon and 0.75 - epsilon < x[1] < 0.75 + epsilon:
        return 3
    elif 0.75 - epsilon < x[0] < 0.75 + epsilon and 0.75 - epsilon < x[1] < 0.75 + epsilon:
        return 3
    elif 0.75 - epsilon < x[0] < 0.75 + epsilon and 0.25 - epsilon < x[1] < 0.25 + epsilon:
        return 3
    elif 0.4 - epsilon < x[0] < 0.4 + epsilon and 0.4 - epsilon < x[1] < 0.4 + epsilon:
        return 1
    elif 0.8 - epsilon < x[0] < 0.8 + epsilon and 0.8 - epsilon < x[1] < 0.8 + epsilon:
        return 2
    else:
        return 4

optimal_cost = 1

x_initial = np.array([0.45, 1.0])
initial_cost = 4

constraints = {
    "equality": [],
    "inequality": [],
    "clip_min": [0, 0],
    "clip_max": [1, 1],
    "categorical": [None, None]
}

def run_test(
        test_id = 4,
        name = "HillClimbing_Radar",
        verbose = True,
        show_graph = True,
        save_graph = True,
        plot_history = False,
        num_samples = 10,

        attack = HillClimbing(estimator=Model4(), verbose=0),
        neighborhood=Radar(constraints),

        targeted=True,
        specific_class=0,
        patience=100,
        cost_function=Cost4,
        static_perturbation_factor=1e-2,
        dynamic_perturbation_factor=1.1,
        enable_negative_inflation_vector=True,
        inflation_vector_max_perturbation=10,
        raise_dynamic_perturbation_after_non_improving_candidate=True,

        # For Simulated Annealing
        initial_temperature=1000,
        final_temperature=1e-10,
        cooling_rate=0.99,

        # For Tabu Search
        tabu_tenure:int=10,
        max_tabu_size:int=100,
        similarity_epsilon:float=1e-6,
):
    initial_time = time.time()
    total_queries = 0
    for _ in range(num_samples):
        if isinstance(attack, HillClimbing):
            _, _, queries = attack.run(input=x_initial,
                                       neighborhood=neighborhood,
                                       targeted=False,
                                       #targeted=targeted,
                                       #specific_class=specific_class,
                                       patience=patience,
                                       cost_function=cost_function,
                                       static_perturbation_factor=static_perturbation_factor,
                                       dynamic_perturbation_factor=dynamic_perturbation_factor,
                                       enable_negative_inflation_vector=enable_negative_inflation_vector,
                                       inflation_vector_max_perturbation=inflation_vector_max_perturbation,
                                       raise_dynamic_perturbation_after_non_improving_candidate=raise_dynamic_perturbation_after_non_improving_candidate,
                                       )

        elif isinstance(attack, SimulatedAnnealing):
            _, _, queries = attack.run(input=x_initial,
                                       neighborhood=neighborhood,
                                       targeted=False,
                                       #targeted=targeted,
                                       #specific_class=specific_class,
                                       patience=patience,
                                       cost_function=cost_function,
                                       static_perturbation_factor=static_perturbation_factor,
                                       dynamic_perturbation_factor=dynamic_perturbation_factor,
                                       enable_negative_inflation_vector=enable_negative_inflation_vector,
                                       inflation_vector_max_perturbation=inflation_vector_max_perturbation,
                                       raise_dynamic_perturbation_after_non_improving_candidate=raise_dynamic_perturbation_after_non_improving_candidate,

                                       initial_temperature=initial_temperature,
                                       final_temperature=final_temperature,
                                       cooling_rate=cooling_rate,
                                       )

        elif isinstance(attack, TabuSearch):
            _, _, queries = attack.run(input=x_initial,
                                       neighborhood=neighborhood,
                                       targeted=False,
                                       #targeted=targeted,
                                       #specific_class=specific_class,
                                       patience=patience,
                                       cost_function=cost_function,
                                       static_perturbation_factor=static_perturbation_factor,
                                       dynamic_perturbation_factor=dynamic_perturbation_factor,
                                       enable_negative_inflation_vector=enable_negative_inflation_vector,
                                       inflation_vector_max_perturbation=inflation_vector_max_perturbation,
                                       raise_dynamic_perturbation_after_non_improving_candidate=raise_dynamic_perturbation_after_non_improving_candidate,

                                       tabu_tenure=tabu_tenure,
                                       max_tabu_size=max_tabu_size,
                                       similarity_epsilon=similarity_epsilon,
                                       )

        else:
            print("Unknown attack.")

    total_queries += queries

    # Plot the maze with color-coding
    plt.imshow(maze[::-1, :], cmap='coolwarm', origin='upper', extent=[0, 1, 0, 1], alpha=0.4)

    # Plot history in a 2D graph with the x[0] and x[1] values
    history = attack.heuristic_history

    for k in range(num_samples):
        x = [x_initial[0]] + [sample[0][0] for sample in history[k]]
        y = [x_initial[1]] + [sample[0][1] for sample in history[k]]

        plt.plot(x, y, marker='o')

    MERob = min([history[k][-1][1] for k in range(num_samples)])
    score = 1 - (MERob - optimal_cost)/(initial_cost - optimal_cost)
    total_time = time.time() - initial_time

    if verbose:
        print("Score:", score)
        print("Total time:", total_time)
        print("Number of queries:", total_queries)

    score_rounded = round(score, 5)
    total_time_rounded = round(total_time, 3)

    if show_graph or save_graph:
        plt.xlabel('x[0]')
        plt.ylabel('x[1]')
        #plt.title(f'{attack.__name__} {neighborhood.__name__}')
        plt.title(name)
        subtitle = ("Score: " + str(score_rounded) + " | Total time: " + str(total_time_rounded) + " | Number of queries: "
                    + str(total_queries))
        plt.suptitle(subtitle)
        if save_graph:
            #plt.savefig(f'graphs/{attack.__name__}_{neighborhood.__name__}_t{test_id}_{score_rounded}_{total_queries}.png')
            plt.savefig(f'graphs/{name}_t{test_id}_{score_rounded}_{total_queries}.png')
        if show_graph:
            plt.show()

    if plot_history:
        # plot evolution of all costs k in history[k][-1][1]
        for k in range(num_samples):
            costs = [sample[1] for sample in history[k]]
            plt.plot(costs)

    return MERob, score, total_time, total_queries