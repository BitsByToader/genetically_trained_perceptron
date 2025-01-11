from perceptron_training import PerceptronTrainingOptimizationProblem
from evolutionary_algorithm import EvolutionaryAlgorithm
import random

if __name__ == '__main__':
    training_algorithm = EvolutionaryAlgorithm()
    training_problem = PerceptronTrainingOptimizationProblem("handwriting_dataset.txt", 0.005, False, [50,50])
    
    winning_chromosome, mean_fitness_report = training_algorithm.solve(training_problem, 50, 5000, 0.9, 0.1)
    print(f'Mean error over training set: {-winning_chromosome.fitness}')
    print(f'Mean error over evaluation set: {training_problem.evaluate_solution(winning_chromosome)}')
    # print(f'Winning chromosome: {winning_chromosome.genes}')

    for i in range(0,10):
        eval_vec = random.choice(training_problem.dataset.evaluation_vectors)
        eval_input = eval_vec[0]
        eval_output = eval_vec[1]
        training_problem.perceptron.compute_output(eval_input)
        print(f'Input: {eval_input}. Expected output: {eval_output}. Computed output: {training_problem.perceptron.output_data}')