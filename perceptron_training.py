from ioptimization_problem import IOptimizationProblem
from chromosome import Chromosome
from dataset import Dataset
from perceptron import Perceptron

class PerceptronTrainingOptimizationProblem(IOptimizationProblem):
    def __init__(self, dataset_path: str, hidden_layer_counts: [int]):
        self.dataset: Dataset = Dataset.from_file(dataset_path)
        self.perceptron: Perceptron = Perceptron.from_counts(self.dataset.input_count, self.dataset.output_count, len(hidden_layer_counts), hidden_layer_counts)
    
    def perceptron_weights_to_chromosome_genes(self) -> [float]:
        weights = self.perceptron.weights
        weights.append(self.perceptron.theta)
        flattened = [
            w
            for wll in weights
            for wl in wll
            for w in wl
        ]
        return flattened

    def chromosome_genes_to_perceptron_weights(self, genes: [float]):
        weights: [[[float]]] = []
        theta: [[float]] = []
        
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

    def compute_fitness(self, chromosome: Chromosome):
        raise NotImplementedError("This method needs to be implemented by a subclass")

    def make_chromosome(self) -> Chromosome:
        gene_count = len(self.perceptron_weights_to_chromosome_genes())
        return Chromosome(gene_count, [-2**31] * gene_count, [2**31-1] * gene_count)