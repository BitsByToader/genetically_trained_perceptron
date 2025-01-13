import tkinter as tk
from perceptron import Perceptron

GRID_SIZE = 32 # 32*32 square
CELL_SIZE = 15  
BRUSH_SIZE = 2  

perceptron = Perceptron()

def toggle_cell(event):
    # change color of cell
    x = event.x // CELL_SIZE
    y = event.y // CELL_SIZE
    if 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE:
        current_color = canvas.itemcget(grid[y][x], 'fill')
        new_color = 'black' if current_color == 'white' else 'white'
        canvas.itemconfig(grid[y][x], fill=new_color)

def draw_continuous(event):
    x = event.x // CELL_SIZE
    y = event.y // CELL_SIZE
    apply_brush(x, y, 'black')

def apply_brush(x, y, color):
    for i in range(-BRUSH_SIZE // 2, BRUSH_SIZE // 2 + 1):
        for j in range(-BRUSH_SIZE // 2, BRUSH_SIZE // 2 + 1):
            xi, yj = x + i, y + j
            if 0 <= xi < GRID_SIZE and 0 <= yj < GRID_SIZE:
                canvas.itemconfig(grid[yj][xi], fill=color)

def clear_grid():
    # all cells reset to white
    for row in grid:
        for cell in row:
            canvas.itemconfig(cell, fill='white')

def get_8x8_matrix()->[int]:
    reduced_matrix = []

    for i in range(0, GRID_SIZE, 4):  
        row = []
        for j in range(0, GRID_SIZE, 4):
            # counting the number of black(1) pixels
            nr_black_pixels = 0
            for x in range(i, i + 4):
                for y in range(j, j + 4):
                    current_color = canvas.itemcget(grid[x][y], 'fill')
                    if current_color == 'black':
                        nr_black_pixels += 1
            
            row.append(nr_black_pixels)
        reduced_matrix.append(row)

    return reduced_matrix

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

def perceptron_from_file(filename: str)-> Perceptron:
    # Process training data from file
    training_data_file = open(filename, "rt")
    
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
    perceptron = Perceptron.from_counts(input_count, output_count, len(hidden_layer_counts), hidden_layer_counts)
    
    apply_training_data_to_perceptron(perceptron, weights)

    return perceptron


def guess():
    #Todo: add logic here
    input_matrix = get_8x8_matrix()
    
    flat_input = [
        x
        for xs in input_matrix
        for x in xs
    ]
    
    normalized_input = [x/16 for x in flat_input]
    perceptron.compute_output(normalized_input)
    computed_output = perceptron.output_data
    digit = computed_output.index(max(computed_output))

    result = str(digit)
    result_window = tk.Toplevel(root)
    result_window.title("Rezultat")
    result_label = tk.Label(result_window, text=result, font=("Arial", 24),  padx=20, pady=20)
    result_label.pack()


root = tk.Tk()
root.title("Paint 2025")

canvas_frame = tk.Frame(root)
canvas_frame.pack(pady=10)

canvas = tk.Canvas(canvas_frame, width=GRID_SIZE * CELL_SIZE, height=GRID_SIZE * CELL_SIZE, bg="white")
canvas.pack()

grid = []
for i in range(GRID_SIZE):
    row = []
    for j in range(GRID_SIZE):
        x1 = j * CELL_SIZE
        y1 = i * CELL_SIZE
        x2 = x1 + CELL_SIZE
        y2 = y1 + CELL_SIZE
        cell = canvas.create_rectangle(x1, y1, x2, y2, fill='white', outline='gray')
        row.append(cell)
    grid.append(row)


canvas.bind("<Button-1>", toggle_cell)  # Simple click for changing the color
canvas.bind("<B1-Motion>", draw_continuous)  # Moving the mouse for continous drawing

button_frame = tk.Frame(root)
button_frame.pack(pady=10)

clear_button = tk.Button(button_frame, text="Clear", command=clear_grid, bg="lightgray", padx=10, pady=5)
clear_button.pack(side=tk.LEFT, padx=5)

guess_button = tk.Button(button_frame, text="Guess", command=guess, bg="lightgray", padx=10, pady=5)
guess_button.pack(side=tk.LEFT, padx=5)

perceptron = perceptron_from_file("training_output.txt")

root.mainloop()
