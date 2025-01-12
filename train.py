from perceptron_training import PerceptronTrainingOptimizationProblem
from evolutionary_algorithm import EvolutionaryAlgorithm

import matplotlib.pyplot as plt
import random

if __name__ == '__main__':
    training_algorithm = EvolutionaryAlgorithm()
    training_problem = PerceptronTrainingOptimizationProblem("handwriting_dataset.txt", 0.005, False, [50,50])
    
    winning_chromosome, mean_fitness_report = training_algorithm.solve(training_problem, 50, 500, 0.9, 0.1)
    print(f'Mean error over training set: {-winning_chromosome.fitness}')
    print(f'Mean error over evaluation set: {training_problem.evaluate_solution(winning_chromosome)}')

    fig, ax = plt.subplots()
    ax.plot(mean_fitness_report)
    plt.show()
    fig.savefig('training_results.pdf', bbox_inches='tight')