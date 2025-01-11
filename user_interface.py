import tkinter as tk

GRID_SIZE = 32 # 32*32 square
CELL_SIZE = 15  
BRUSH_SIZE = 2  

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

def guess():
    result = "N/A"
    #Todo: add logic here
    
    result_window = tk.Toplevel(root)
    result_window.title("Rezultat")
    result_label = tk.Label(result_window, text=result, font=("Arial", 24), bg="lightblue", padx=20, pady=20)
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
canvas.bind("<B1-Motion>", draw_continuous)  # moving the mouse for continous drawing

button_frame = tk.Frame(root)
button_frame.pack(pady=10)

clear_button = tk.Button(button_frame, text="Clear", command=clear_grid, bg="lightgray", padx=10, pady=5)
clear_button.pack(side=tk.LEFT, padx=5)

guess_button = tk.Button(button_frame, text="Guess", command=guess, bg="lightgray", padx=10, pady=5)
guess_button.pack(side=tk.LEFT, padx=5)

root.mainloop()
