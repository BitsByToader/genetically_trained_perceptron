from random import shuffle


# Class is responsible for loading and storing a training/evaluation dataset for a neural network.
class Dataset:
    # Constructs an empty dataset
    def __init__(self):
        self.dataset_path: str = ""
        self.input_count: int = 0
        self.output_count: int = 0
        
        # Element of list will be a list containing two elements:
        #   1. list of input data;
        #   2. list of corresponding output data.
        # The two lists will only contain floats.
        self.dataset_vectors: [[[float]]] = []

        # Same as above
        self.training_vectors: [[[float]]] = []
        self.evaluation_vectors: [[[float]]] = []

    # Constructs a dataset based on a given file found at dataset_path.
    @staticmethod
    def from_file(dataset_path: str):
        cls = Dataset()
        cls.dataset_path = dataset_path
        cls.process_dataset_file()
        return cls

    # Processes the dataset file found at dataset_path.
    def process_dataset_file(self):
        dataset_file = open(self.dataset_path)
        
        dataset_structure_str = dataset_file.readline()
        dataset_structure = dataset_structure_str.split(',')
        self.input_count = int(dataset_structure[0])
        self.output_count = int(dataset_structure[1])

        global_min = 0.1
        global_max = 1.0

        for vector_str in dataset_file:
            # Dataset line beginning with # is a comment.
            if len(vector_str) == 0 or vector_str[0] == '#':
                continue

            input_output_str_list = vector_str.split(',')

            if len(input_output_str_list) != (self.input_count + self.output_count):
                raise Exception('Number of values doesnt match for training vector!')
            
            input_output_list = [float(i) for i in input_output_str_list]
            input_list = input_output_list[0:(self.input_count)]
            output_list = input_output_list[(self.input_count):]

            local_min = min(input_list)
            local_max = max(input_list)

            global_max = local_max if local_max > global_max else global_max
            global_min = local_min if local_min < global_min else global_min

            dataset_vector = [input_list, output_list]
            self.dataset_vectors.append(dataset_vector)

        dataset_file.close()

        # Apply min max normalization to dataset
        divisor = global_max - global_min
        for vector_idx in range(len(self.dataset_vectors)):
            for attribute_idx in range(len(self.dataset_vectors[vector_idx][0])):
                self.dataset_vectors[vector_idx][0][attribute_idx] = (self.dataset_vectors[vector_idx][0][attribute_idx] - global_min) / divisor

    # Given a rate of evaluation vectors to training vectors, shuffles the dataset and splits it into a list
    # of training vectors and a list of evaluation vectors.
    def split_dataset_vectors(self, evaluation_to_training_rate: float):
        shuffle(self.dataset_vectors)
        training_vector_count: int = int(evaluation_to_training_rate * float(len(self.dataset_vectors)))
        
        self.training_vectors = self.dataset_vectors[0:training_vector_count]
        self.evaluation_vectors = self.dataset_vectors[training_vector_count:]