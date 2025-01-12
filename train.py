from perceptron_training import PerceptronTrainingOptimizationProblem
from evolutionary_algorithm import EvolutionaryAlgorithm

import matplotlib.pyplot as plt
import random

if __name__ == '__main__':
    # Initialize problem
    training_algorithm = EvolutionaryAlgorithm()
    training_problem = PerceptronTrainingOptimizationProblem("handwriting_dataset.txt", 0.01, False, [50,50])
    
    # Train perceptron
    winning_chromosome, mean_fitness_report = training_algorithm.solve(training_problem, 50, 3000, 0.9, 0.1)
    print(f'Mean error over training set: {-winning_chromosome.fitness}')
    print(f'Mean error over evaluation set: {training_problem.evaluate_solution(winning_chromosome)}')

    # Save winning perceptron to file
    chromosome_file = open("training_output.txt", "wt")
    # Header describing the structure of the trained perceptron: input_count, output_count and list of neuron count per hidden layers.
    chromosome_file.write(f'{training_problem.perceptron.input_count},{training_problem.perceptron.output_count},{",".join([str(i) for i in training_problem.perceptron.neurons_per_hidden_layer])}\n')
    chromosome_file.write(','.join([str(f) for f in winning_chromosome.genes]))
    chromosome_file.close()

    # Plot, display, and save results.
    fig, ax = plt.subplots()
    ax.plot(mean_fitness_report)
    plt.show()
    fig.savefig('training_results.pdf', bbox_inches='tight')