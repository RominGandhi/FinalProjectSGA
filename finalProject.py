import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import messagebox, simpledialog

# Objective functions
def sphere_function(x):
    return np.sum(x**2)

def rosenbrock_function(x):
    return np.sum(100 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)

def himmelblau_function(x):
    return (x[0]**2 + x[1] - 11)**2 + (x[0] + x[1]**2 - 7)**2

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

    return best_individual, best_fitness

# Tkinter GUI
def run_gui():
    def start_ga():
        # Retrieve inputs
        try:
            objective_choice = int(obj_func_var.get())
            if objective_choice == 1:
                objective_function = sphere_function
            elif objective_choice == 2:
                objective_function = rosenbrock_function
            elif objective_choice == 3:
                objective_function = himmelblau_function
            elif objective_choice == 4:
                func_input = custom_func_entry.get()
                objective_function = eval(f"lambda x: {func_input}")
            else:
                messagebox.showerror("Error", "Invalid objective function choice.")
                return

            num_variables = int(num_vars_entry.get())
            bounds = []
            for i in range(num_variables):
                bounds.append(tuple(map(float, bounds_entries[i].get().split(','))))

            population_size = int(pop_size_entry.get())
            generations = int(generations_entry.get())
            mutation_rate = float(mutation_rate_entry.get())

            # Run GA
            best_solution, best_fitness  = genetic_algorithm(objective_function, bounds, population_size, generations, mutation_rate)
            messagebox.showinfo(
            "Best Solution",
            f"Best Solution Found: {best_solution}\nFinal Fitness Score: {best_fitness}"
        )
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {e}")

    # Tkinter window setup
    root = tk.Tk()
    root.title("Genetic Algorithm GUI")

    # Objective function selection
    tk.Label(root, text="Select Objective Function").grid(row=0, column=0, columnspan=2)
    obj_func_var = tk.StringVar(value="1")
    tk.Radiobutton(root, text="Sphere Function", variable=obj_func_var, value="1").grid(row=1, column=0)
    tk.Radiobutton(root, text="Rosenbrock Function", variable=obj_func_var, value="2").grid(row=2, column=0)
    tk.Radiobutton(root, text="Himmelblau Function", variable=obj_func_var, value="3").grid(row=3, column=0)
    tk.Radiobutton(root, text="Custom Function", variable=obj_func_var, value="4").grid(row=4, column=0)
    custom_func_entry = tk.Entry(root, width=40)
    custom_func_entry.grid(row=4, column=1)

    # Number of variables
    tk.Label(root, text="Number of Variables:").grid(row=5, column=0)
    num_vars_entry = tk.Entry(root)
    num_vars_entry.grid(row=5, column=1)

    # Bounds for each variable
    tk.Label(root, text="Enter Bounds (min,max for each variable):").grid(row=6, column=0, columnspan=2)
    bounds_entries = [tk.Entry(root) for _ in range(10)]  # Assume max 10 variables for simplicity
    for i, entry in enumerate(bounds_entries):
        entry.grid(row=7+i, column=0, columnspan=2)

    # Population size, generations, mutation rate
    tk.Label(root, text="Population Size:").grid(row=17, column=0)
    pop_size_entry = tk.Entry(root)
    pop_size_entry.insert(0, "50")
    pop_size_entry.grid(row=17, column=1)

    tk.Label(root, text="Generations:").grid(row=18, column=0)
    generations_entry = tk.Entry(root)
    generations_entry.insert(0, "100")
    generations_entry.grid(row=18, column=1)

    tk.Label(root, text="Mutation Rate:").grid(row=19, column=0)
    mutation_rate_entry = tk.Entry(root)
    mutation_rate_entry.insert(0, "0.1")
    mutation_rate_entry.grid(row=19, column=1)

    # Start button
    tk.Button(root, text="Start Genetic Algorithm", command=start_ga).grid(row=20, column=0, columnspan=2)

    root.mainloop()

# Run the GUI
if __name__ == "__main__":
    run_gui()
