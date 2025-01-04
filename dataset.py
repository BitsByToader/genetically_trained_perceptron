# Class is responsible for loading and storing a training/evaluation dataset for a neural network.
class Dataset:
    # Constructs an empty dataset
    def __init__(self):
        self.dataset_path = ""
        self.input_count = 0
        self.output_count = 0
        
        # Element of list will be a list containing two elements:
        #   1. list of input data;
        #   2. list of corresponding output data.
        # The two lists will only contain floats.
        self.dataset_vectors = []


    # Constructs a dataset based on a given file found at dataset_path.
    def __init__(self, dataset_path: str):
        self.__init__()
        self.dataset_path = dataset_path
        self.process_dataset_file()

    # Processes the dataset file found at dataset_path.
    def process_dataset_file(self):
        dataset_file = open(dataset_path)
        
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