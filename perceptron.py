import math 
import numpy as np

class Perceptron:
    # Constructs an empty perceptron
    def __init__(self):
        self.input_count: int = 0
        self.output_count: int = 0
        self.hidden_layers_count: int = 0
        self.neurons_per_hidden_layer: [int] = []
        self.output_data: [float] = []
        self.weights: [[[float]]] = []
        self.theta: [[float]] = []

    # Constructs a perceptron with given information
    @staticmethod
    def from_counts(input_count: int, output_count: int, hidden_layers_count: int, neurons_per_hidden_layer: [int]) -> 'Perceptron':
        perceptron = Perceptron()
        perceptron.input_count = input_count
        perceptron.output_count = output_count
        perceptron.hidden_layers_count = hidden_layers_count
        perceptron.neurons_per_hidden_layer = neurons_per_hidden_layer
        
        perceptron.weight_initialization()
        perceptron.theta_initialization()
        return perceptron
    
    @staticmethod
    def from_weights(input_count: int, output_count: int, hidden_layers_count: int, neurons_per_hidden_layer: [int], weights: [[[float]]], theta: [[float]]) -> 'Perceptron':
        perceptron = Perceptron()
        perceptron.input_count = input_count
        perceptron.output_count = output_count
        perceptron.hidden_layers_count = hidden_layers_count
        perceptron.neurons_per_hidden_layer = neurons_per_hidden_layer

        perceptron.weights = weights
        perceptron.theta = theta
        return perceptron

    # Initialize weights according to the available data
    def weight_initialization(self):
        neuron_per_layers = [self.input_count] + self.neurons_per_hidden_layer + [self.output_count]
        print("nr neurons per layer: ",neuron_per_layers)

        for i in range(1,len(neuron_per_layers)):
            a = []
            for j in range(neuron_per_layers[i]):
                b = [0.0] * neuron_per_layers[i - 1]
                a.append(b)
            self.weights.append(a)
    
    # Initialize theta according to the available data
    def theta_initialization(self):
        neuron_per_layers = [self.input_count] + self.neurons_per_hidden_layer + [self.output_count]

        for i in range(1,len(neuron_per_layers)):
            b = [0.0] * neuron_per_layers[i]
            self.theta.append(b)

    # Given an input the perceptron calculates the output
    def compute_output(self, input: [float]):
        if (len(input) != self.input_count):
            raise Exception("Input length doesn't match input_count")
        
        actual_output = input.copy()
        for layer in range(len(self.weights)):
            inner_input = [1.0] * len(self.weights[layer])
            for i in range(len(self.weights[layer])):    
                for j in range(len(self.weights[layer][i])):
                    inner_input[i] *= actual_output[j] * self.weights[layer][i][j]
                
                inner_input[i] -= self.theta[layer][i]
                inner_input[i] = self.sigmoid_activation(inner_input[i]) # Activation function
            
            actual_output = inner_input.copy()

        self.output_data = actual_output.copy()

    # Bipolar sigmoid activation function
    @staticmethod
    def sigmoid_activation(x: float) -> float: # TODO: Change to name to bipolar_sigmoid_activation
        return (1 - math.exp(-2*x))/(1 + math.exp(-2*x))

    @staticmethod
    def stable_sigmoid(x: float) -> float:
        # TODO: replace with standard library implementation
        return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))


    # Calculates the mean square error
    def compute_error(self, desired_output: [float]):
        if (len(desired_output) != self.output_count):
            raise Exception("Desired_output doesn't match output_count")
        
        err = [0.0] * self.output_count 
        mean_error = 0.0
        
        for i in range(self.output_count):
            err[i] = (desired_output[i] - self.output_data[i]) * (desired_output[i] - self.output_data[i])
            mean_error += err[i]
        
        mean_error /= len(err)
        return mean_error
