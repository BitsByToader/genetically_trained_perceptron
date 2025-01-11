import math 

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

    # Bipolar sigmoid activation function
    @staticmethod
    def bipolar_sigmoid_activation(x: float) -> float:
        # Bind input as to not overflow (TODO: correct?)
        # if x >= 30.0:
        #     x = 30.0
        
        # if x <= -30.0:
        #     x = -30.0
        
        # Compute bipolar sigmoid
        return (1.0 - math.exp(-2*x))/(1.0 + math.exp(-2*x))

    # Numerically stable sigmoid activation function
    @staticmethod
    def stable_sigmoid_activation(x: float) -> float:
        if x >= 0:
            return 1. / ( 1. + math.exp(-x) )
        else:
            return math.exp(x) / ( 1. + math.exp(x) )

    # Step activation function
    @staticmethod
    def step_activation(x: float) -> float:
        return (1.0 if x >= 0 else 0.0)

    # Rectified Linear Unit activation function
    @staticmethod
    def relu_activation(x: float) -> float:
        return (x if x > 0 else 0.0)

    @staticmethod
    def softmax(inputs: [float]) -> [float]:
        max_input = max(inputs)
        inputs_bounded = [x-max_input for x in inputs]
        inputs_exp: [float] = [math.exp(x) for x in inputs_bounded ]
        inputs_sum: float = math.fsum(inputs_exp)
        return [x / inputs_sum for x in inputs_exp]

    # Given an input the perceptron calculates the output
    def compute_output(self, input: [float]):
        if (len(input) != self.input_count):
            raise Exception("Input length doesn't match input_count")
        
        previous_layer = input.copy()
        for layer in range(len(self.weights)):
            neuron_values = [1.0] * len(self.weights[layer])
            for neuron_idx in range(len(self.weights[layer])):
                for neuron_input_idx in range(len(self.weights[layer][neuron_idx])):
                    neuron_values[neuron_idx] += previous_layer[neuron_input_idx] * self.weights[layer][neuron_idx][neuron_input_idx]
                
                neuron_values[neuron_idx] -= self.theta[layer][neuron_idx]
                if layer < len(self.weights)-1:
                    # Apply regular activation function to all layers besides output.
                    # Output will use softmax function as 'activation' function.
                    neuron_values[neuron_idx] = self.relu_activation(neuron_values[neuron_idx])
            
            previous_layer = neuron_values.copy()

        self.output_data = self.softmax(previous_layer)

    # Calculates the mean square error
    # def compute_error(self, desired_output: [float]) -> float:
    #     if (len(desired_output) != self.output_count):
    #         raise Exception("Desired_output doesn't match output_count")
        
    #     err = [0.0] * self.output_count 
    #     mean_error = 0.0
        
    #     for i in range(self.output_count):
    #         err[i] = (desired_output[i] - self.output_data[i]) * (desired_output[i] - self.output_data[i])
    #         mean_error += err[i]
        
    #     mean_error /= len(err)
    #     return mean_error

    # Calculates the categorical loss entropy of the computed output data and the expected class.
    # The method expects that the desired and computed output have the classes encoded as one-hot.
    def compute_error(self, desired_output: [float]) -> float:
        epsilon = 1e-7  # Small value to avoid log(0)
        predicted_probs = [max(p, epsilon) for p in self.output_data]  # Ensure no zero probabilities
        loss = -sum(t * math.log(p) for t, p in zip(desired_output, self.output_data))
        return loss