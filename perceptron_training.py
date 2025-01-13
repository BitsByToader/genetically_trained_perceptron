from ioptimization_problem import IOptimizationProblem
from chromosome import Chromosome
from dataset import Dataset
from perceptron import Perceptron

from multiprocessing import Pool

class PerceptronTrainingOptimizationProblem(IOptimizationProblem):
    def __init__(self, pool: Pool, dataset_path: str, dataset_split_rate: float, normalize_dataset: bool, hidden_layer_counts: [int]):
        self.pool = pool
        
        self.dataset: Dataset = Dataset.from_file(dataset_path, normalize_dataset)
        self.dataset.split_dataset_vectors(dataset_split_rate)
        self.perceptron: Perceptron = Perceptron.from_counts(self.dataset.input_count, self.dataset.output_count, len(hidden_layer_counts), hidden_layer_counts)
        self.gene_count = len(self.perceptron_weights_to_chromosome_genes())
    
    def perceptron_weights_to_chromosome_genes(self) -> [float]:
        weights = self.perceptron.weights.copy()
        weights.append(self.perceptron.theta)
        flattened = [
            w
            for wll in weights
            for wl in wll
            for w in wl
        ]
        return flattened

    def apply_chromosome_to_perceptron(self, chromosome: Chromosome):
        weights: [[[float]]] = []
        theta: [[float]] = []
        genes = list(chromosome.genes)

        neuron_count_per_layer = [self.perceptron.input_count] + self.perceptron.neurons_per_hidden_layer + [self.perceptron.output_count]
        for idx in range(1, len(neuron_count_per_layer)):
            curr_layer_neuron_count = neuron_count_per_layer[idx]
            prev_layer_neuron_count = neuron_count_per_layer[idx-1]

            layer_weights: [[float]] = []
            for _ in range(0,curr_layer_neuron_count):
                layer_weights.append(genes[:prev_layer_neuron_count])
                del genes[:prev_layer_neuron_count]
            
            weights.append(layer_weights)

        for idx in range(1, len(neuron_count_per_layer)):
            curr_layer_neuron_count = neuron_count_per_layer[idx]
            
            layer_thresholds: [float] = genes[:curr_layer_neuron_count]
            del genes[:curr_layer_neuron_count]
            
            theta.append(layer_thresholds)

        self.perceptron.weights = weights
        self.perceptron.theta = theta

    @staticmethod
    def compute_error(args) -> float:
        perceptron, training_vector = args
        
        input_data: [float] = training_vector[0]
        output_data: [float] = training_vector[1]

        perceptron.compute_output(input_data)
        return perceptron.compute_error(output_data)

    def evaluate_solution(self, solution: Chromosome) -> float:
        error: float = 0.0

        self.apply_chromosome_to_perceptron(solution)

        for vector in self.dataset.evaluation_vectors:
            error += self.compute_error(self.perceptron, vector)

        error = error / len(self.dataset.evaluation_vectors)
        return error

    def compute_fitness(self, chromosome: Chromosome):
        error: float = 0.0

        # Apply chromosome genes to perceptron to compute error.
        self.apply_chromosome_to_perceptron(chromosome)

        result = self.pool.map(self.compute_error, [(self.perceptron, v) for v in self.dataset.training_vectors], 50)
        error = sum(result) / len(self.dataset.training_vectors)
        
        # Negate the error because the algorithm maximizes the fitness, whereas we want to minimize the error.
        chromosome.fitness = -error

        # print(f'Computed fitness for a chromosome: {chromosome.fitness}')

    def make_chromosome(self) -> Chromosome:
        return Chromosome(self.gene_count, [-1.0] * self.gene_count, [1.0] * self.gene_count)
