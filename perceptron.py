class Perceptron:
    def __init__(self, input, hidden_layers, nr_output):
        self.input = input
        self.hidden_layers = hidden_layers #a list that contains nr of neurons per hidden layer
        self.nr_output = nr_output

    def weight_initialization(self):
        layers = [len(self.input)] + self.hidden_layers + [self.nr_output]
        print(layers)
        self.weights = []
        self.theta = []

        for i in range(1,len(layers)):
            a = []
            for j in range(layers[i]):
                b = [0.0] * layers[i - 1]
                a.append(b)
            self.weights.append(a)
        
        for i in range(1,len(layers)):
            b = [-1.0] * layers[i]
            self.theta.append(b)

    def compute_output(self):
        x  = self.input
        for layer in range(len(self.weights)):
            inner_input = [1.0] * len(self.weights[layer])
            for i in range(len(self.weights[layer])):    
                for j in range(len(self.weights[layer][i])):
                    inner_input[i] *= x[j] * self.weights[layer][i][j]
                inner_input[i] -= self.theta[layer][i]
            x = inner_input
            print(x)
        self.actual_output = x

    def compute_error(self, desired_output):
        err = [0.0] * len(self.actual_output)   
        mean_error = 0.0
        for i in range(len(self.actual_output)):
            err[i] = (1/2) * (self.actual_output[i] - desired_output[i]) * (self.actual_output[i] - desired_output[i])
            mean_error+=err[i]
        mean_error /= len(err)
    
if __name__ == '__main__':
    perceptron = Perceptron([1, 1, 1, 1, 1], [3, 2, 3], 4)
    perceptron.weight_initialization()
    print(perceptron.weights)
    print(perceptron.theta)
    perceptron.compute_output()
    perceptron.compute_error([0, 0, 0, 0])
