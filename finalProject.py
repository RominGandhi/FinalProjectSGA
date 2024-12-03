import numpy as np
import matplotlib.pyplot as plt

# Objective functions
def sphere_function(x):
    return np.sum(x**2)

def rosenbrock_function(x):
    return np.sum(100 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)

def himmelblau_function(x):
    return (x[0]**2 + x[1] - 11)**2 + (x[0] + x[1]**2 - 7)**2

#######################################################################################
# genetic_algorithm() - genetic algorithm used to minimize an objective function
#                       using reproduction, crossover, and mutation
#
# Params: objective_function (function), bounds (list of tuples), population_size (int) 
#         generations (int), mutation_rate (float)
#
# Return: best_individual (ndarray)
#
def genetic_algorithm(
    objective_function, bounds, population_size=50, generations=100, mutation_rate=0.1
):
    # Number of variables in problem
    dim = len(bounds)

    # Create population with random variables within bounds
    population = np.random.uniform(
        [b[0] for b in bounds],  # Lower bounds
        [b[1] for b in bounds],  # Upper bounds
        (population_size, dim)   # Shape of population 
    )

    num_elites = 4
    best_fitness_per_generation = []

    # Iterate through each generation
    for generation in range(generations):
        # Obtain fitness of every individual
        fitness = np.array([objective_function(ind) for ind in population])

        # Keep track of elites in the generation
        elite_indices = np.argsort(fitness)[:num_elites]
        elite_individuals = population[elite_indices]

        # Calculate probabilities for biased wheel
        total_fitness = np.sum(fitness) 
        selection_probs = (1 / (fitness + 1e-6)) / total_fitness  

        # Normalize (ensure probabilities add to 1)
        selection_probs /= np.sum(selection_probs)

        # Select mating pool based on biased wheel
        selected = population[np.random.choice(len(population), size=population_size, p=selection_probs)]

        # Create offspring using crossover and mutation
        offspring = []
        for _ in range((population_size - num_elites) // 2):  
            # Randomly select 2 parents from mating pool
            p1, p2 = selected[np.random.choice(len(selected), size=2)]

            # Create crossover point and make children based on parents and crossover point
            crossover_point = np.random.randint(1, dim)
            child1 = np.concatenate((p1[:crossover_point], p2[crossover_point:]))
            child2 = np.concatenate((p2[:crossover_point], p1[crossover_point:]))
            
            # Chance to mutate a child
            if np.random.rand() < mutation_rate:
                child1 += np.random.uniform(-0.1, 0.1, dim)
            if np.random.rand() < mutation_rate:
                child2 += np.random.uniform(-0.1, 0.1, dim)
            
            # Add children to list
            offspring.append(child1)
            offspring.append(child2)

        # Replace old population with elites and children
        population = np.vstack((elite_individuals, np.array(offspring)))


        # Print generation and number of individuals
        print(f"Generation {generation + 1} - Number of Individuals: {len(population)}")

        # Obtain best individual and fitness in current generation
        best_individual_gen = population[np.argmin([objective_function(ind) for ind in population])]
        best_fitness_gen = objective_function(best_individual_gen)
        best_fitness_per_generation.append(best_fitness_gen)

        # Print best individual and fitness score in each generation
        print(f"Generation {generation + 1} - Best Solution: {best_individual_gen}, Fitness: {best_fitness_gen}")

    # Obtain best overall individual and fitness score
    best_individual = population[np.argmin([objective_function(ind) for ind in population])]
    best_fitness = objective_function(best_individual)

    # Print overall best individual and fitness score
    print("\nFinal Best Solution:", best_individual)
    print("Final Fitness Score:", best_fitness)

    # Plot the fitness over generations
    plt.plot(range(1, generations + 1), best_fitness_per_generation)
    plt.xlabel("Generation")
    plt.ylabel("Best Fitness")
    plt.title("Best Fitness Over Generations")
    plt.grid(True)
    plt.show()

    return best_individual


# Create bounds and run the GA
bounds = [(-5, 5)] * 2  # Bounds for 2D
solution = genetic_algorithm(rosenbrock_function, bounds)