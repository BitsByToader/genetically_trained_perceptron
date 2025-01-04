import math 

class Perceptron:
    # Constructs an empty perceptron
    def __init__(self):
        self.input_count: int = 0
        self.output_count: int = 0
        self.hidden_layers_count: int = 0
        self.neutrons_per_layer: [int] = []
        self.output_data: [float] = []
        self.weights: [[[float]]]= []
        self.theta: [[float]] = []

    # Constructs a perceptron with given information
    @staticmethod
    def from_counts(input_count: int, output_count: int, hidden_layers_count: int, neutrons_per_layer: list):
        perceptron = Perceptron()
        perceptron.input_count = input_count
        perceptron.output_count = output_count
        perceptron.hidden_layers_count = hidden_layers_count
        perceptron.neutrons_per_layer = neutrons_per_layer
        
        perceptron.weight_initialization()
        perceptron.theta_initialization()
        return perceptron
    
    @staticmethod
    def from_file(weights: [[[float]]]):
        perceptron = Perceptron()
        perceptron.weights = weights

    # Initialize weights according to the available data
    def weight_initialization(self):
        layers = [self.input_count] + self.neutrons_per_layer + [self.output_count]
        print("nr neurons per layer: ",layers)

        for i in range(1,len(layers)):
            a = []
            for j in range(layers[i]):
                b = [0.0] * layers[i - 1]
                a.append(b)
            self.weights.append(a)
    
    # Initialize theta according to the available data
    def theta_initialization(self):
        layers = [self.input_count] + self.neutrons_per_layer + [self.output_count]

        for i in range(1,len(layers)):
            b = [0.0] * layers[i]
            self.theta.append(b)

    # Given an input the perceptron calculates the output
    def compute_output(self, input: list):
        actual_output = input
        
        for layer in range(len(self.weights)):
            inner_input = [1.0] * len(self.weights[layer])
            for i in range(len(self.weights[layer])):    
                for j in range(len(self.weights[layer][i])):
                    inner_input[i] *= actual_output[j] * self.weights[layer][i][j]
                
                inner_input[i] -= self.theta[layer][i]
                inner_input[i] = self.sigmoid_activation(inner_input[i]) # Activation function
            
            actual_output = inner_input

        self.output_data = actual_output
        print("Actual output: ",self.output_data)

    # Bipolar sigmoid activation function
    @staticmethod
    def sigmoid_activation(x: float):
        return (1 - math.exp(-2*x))/(1 + math.exp(-2*x))
    
    # Calculates the mean square error
    def compute_error(self, desired_output: list):
        err = [0.0] * self.output_count 
        mean_error = 0.0
        for i in range(self.output_count):
            err[i] = (desired_output[i] - self.output_data[i]) * (desired_output[i] - self.output_data[i])
            mean_error += err[i]
        mean_error /= len(err)
        print("Mean error: ",mean_error)
    
if __name__ == '__main__':
    perceptron = Perceptron.from_counts(5, 4, 3, [3, 2, 3])
    print("weights: ",perceptron.weights)
    print("theta: ",perceptron.theta)
    perceptron.compute_output([1, 1, 1, 1, 1])
    perceptron.compute_error([0, 0, 0, 0])
