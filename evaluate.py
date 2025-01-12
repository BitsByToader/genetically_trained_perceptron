from dataset import Dataset
from perceptron import Perceptron


def apply_training_data_to_perceptron(perceptron: Perceptron, training_data: [float]):
    weights: [[[float]]] = []
    theta: [[float]] = []
    data = training_data

    neuron_count_per_layer = [perceptron.input_count] + perceptron.neurons_per_hidden_layer + [perceptron.output_count]
    for idx in range(1, len(neuron_count_per_layer)):
        curr_layer_neuron_count = neuron_count_per_layer[idx]
        prev_layer_neuron_count = neuron_count_per_layer[idx-1]

        layer_weights: [[float]] = []
        for _ in range(0,curr_layer_neuron_count):
            layer_weights.append(data[:prev_layer_neuron_count])
            del data[:prev_layer_neuron_count]
        
        weights.append(layer_weights)

    for idx in range(1, len(neuron_count_per_layer)):
        curr_layer_neuron_count = neuron_count_per_layer[idx]
        
        layer_thresholds: [float] = data[:curr_layer_neuron_count]
        del data[:curr_layer_neuron_count]
        
        theta.append(layer_thresholds)

    perceptron.weights = weights
    perceptron.theta = theta


if __name__ == '__main__':
    # Process training data from file
    training_data_file = open("training_output.txt", "rt")
    
    # Read perceptron structure
    header_str = training_data_file.readline()
    header_data = [int(s) for s in header_str.split(',')]
    input_count = header_data[0]
    output_count = header_data[1]
    hidden_layer_counts = header_data[2:]

    # Read perceptron weights
    weights_str = training_data_file.readline()
    weights = [float(s) for s in weights_str.split(',')]

    training_data_file.close()

    # Create perceptron and apply weights
    perceptron = Perceptron.from_counts(input_count, output_count, len(hidden_layer_counts), hidden_layer_counts)
    apply_training_data_to_perceptron(perceptron, weights)

    # Open and process dataset.
    dataset = Dataset.from_file("handwriting_dataset.txt", 0)

    # Evaluate perceptron using dataset
    for i in range(10):
        vector = dataset.dataset_vectors[i]
        vinput = vector[0]
        voutput = vector[1]
        expected_class = voutput.index(max(voutput))

        perceptron.compute_output(vinput)
        comp_output = perceptron.output_data
        comp_class = comp_output.index(max(comp_output))

        print(f'Input data: {vinput}')
        print(f'Expected output: {voutput}')
        print(f'Computed output: {comp_output}')
        print(f'Expected class: {expected_class}')
        print(f'Computed class: {comp_class}')
        print()