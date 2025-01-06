from perceptron_training import PerceptronTrainingOptimizationProblem
from evolutionary_algorithm import EvolutionaryAlgorithm

# TODO: Improve printing

if __name__ == '__main__':
    training_algorithm = EvolutionaryAlgorithm()
    training_problem = PerceptronTrainingOptimizationProblem("xor_dataset.txt", [4,2])
    
    winning_chromosome, mean_fitness_report = training_algorithm.solve(training_problem, 10, 50, 0.9, 0.1)
    print(f'Mean error over training set: {-winning_chromosome.fitness}')
    #print(f'Mean error over evaluation set: {training_problem.evaluate_solution(winning_chromosome)}')

    print(f'Mean errors over generations: {mean_fitness_report}')

    #training_problem.perceptron.compute_output([6.2,2.8,4.8,1.8])
    #print(f'Some output for some input: {training_problem.perceptron.output_data}')
