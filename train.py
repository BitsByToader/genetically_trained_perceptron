from perceptron_training import PerceptronTrainingOptimizationProblem

# TODO: Improve printing

if __name__ == '__main__':
    training_problem = PerceptronTrainingOptimizationProblem("/Users/tudor/Documents/proiect_ia/data_sets/iris/iris_dataset.txt", [2, 5, 3])
    print(training_problem.perceptron.weights)
    print(training_problem.perceptron.theta)

    genes = training_problem.perceptron_weights_to_chromosome_genes()
    training_problem.chromosome_genes_to_perceptron_weights(genes)