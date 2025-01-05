from perceptron_training import PerceptronTrainingOptimizationProblem
from evolutionary_algorithm import EvolutionaryAlgorithm

# TODO: Improve printing

if __name__ == '__main__':
    training_algorithm = EvolutionaryAlgorithm()
    training_problem = PerceptronTrainingOptimizationProblem("/Users/tudor/Documents/proiect_ia/data_sets/iris/iris_dataset.txt", [2, 5, 3])

    winning_chromosome = training_algorithm.solve(training_problem, 10, 100, 0.9, 0.1)
    print(f'Mean error over training set: {-winning_chromosome.fitness}')