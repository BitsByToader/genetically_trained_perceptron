from perceptron_training import PerceptronTrainingOptimizationProblem
from evolutionary_algorithm import EvolutionaryAlgorithm
import math 
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

# TODO: Improve printing
    

if __name__ == '__main__':
    training_algorithm = EvolutionaryAlgorithm()
    training_problem = PerceptronTrainingOptimizationProblem("/root/proiectIA/iris_dataset.txt", [2, 5, 3])
    
    winning_chromosome, mean_fitness_report = training_algorithm.solve(training_problem, 100, 10, 0.9, 0.1)
    print(f'Mean error over training set: {-winning_chromosome.fitness}')
    print(f'Mean error over evaluation set: {training_problem.evaluate_solution(winning_chromosome)}')

    training_problem.perceptron.compute_output([6.2,2.8,4.8,1.8])
    print(f'Some output for some input: {training_problem.perceptron.output_data}')
    print("Meanreport",mean_fitness_report)
    print(mean_fitness_report)