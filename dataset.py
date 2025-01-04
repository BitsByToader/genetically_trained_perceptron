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
        self.dataset_vectors: [[float]] = []

        # Same as above
        self.training_vectors: [[float]] = []
        self.evaluation_vectors: [[float]] = []


    # Constructs a dataset based on a given file found at dataset_path.
    def __init__(self, dataset_path: str):
        self.__init__()
        self.dataset_path = dataset_path
        self.process_dataset_file()

    # Processes the dataset file found at dataset_path.
    def process_dataset_file(self):
        dataset_file = open(self.dataset_path)
        
        dataset_structure_str = dataset_file.readLine()
        dataset_structure = dataset_structure_str.decode("utf-8").split(',')
        self.input_count = int(dataset_structure[0])
        self.output_count = int(dataset_structure[1])

        for vector_str in dataset_file:
            input_output_str_list = vector_str.decode("utf-8").split(',')
            
            if len(input_output_str_list) != (self.input_count + self.output_count):
                raise Exception('Number of values doesnt match for training vector!')
            
            input_output_list = [float(i) for i in input_output_str_list]
            input_list = input_output_list[0:(self.input_count)]
            output_list = input_output_list[(self.input_count):]

            dataset_vector = [input_list, output_list]
            self.dataset_vectors.append(dataset_vector)

        f.close()

    # Given a rate of evaluation vectors to training vectors, shuffles the dataset and splits it into a list
    # of training vectors and a list of evaluation vectors.
    def split_dataset_vectors(self, evaluation_to_training_rate: float):
        shuffle(self.dataset_vectors)
        training_vector_count: int = int(evaluation_to_training_rate * float(len(self.dataset_vectors)))
        
        self.training_vectors = self.dataset_vectors[0:training_vector_count]
        self.evaluation_vectors = self.dataset_vectors[training_vector_count:]