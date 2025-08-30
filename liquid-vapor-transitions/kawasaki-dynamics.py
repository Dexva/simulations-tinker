import tkinter as tk
import time
import numpy as np
import random
import math

# --- Grid Constants ---
GRID_WIDTH = 20     # Number of cells horizontally
GRID_HEIGHT = 20      # Number of cells vertically
CELL_SIZE = 20        # Size of each cell in pixels
SCREEN_WIDTH = GRID_WIDTH * CELL_SIZE
SCREEN_HEIGHT = GRID_HEIGHT * CELL_SIZE

# --- Simulation Constants ---
DENSITY = 0.3
TEMP_INIT = 0.01
TEMP_MAX = 1.0
TEMP_MIN = 0.01

# Colors
WHITE = "#ffffff" 
BLACK = "#000000"
GRID_LINE_COLOR = "#c8c8c8" 
CELL_COLOR = "#34cfeb" 
KNOB_COLOR = "#e63946"

class Coordinate:
    def __init__(self, x_coord, y_coord):
        self.x = x_coord
        self.y = y_coord

def populate():
    p = random.random()
    if p <= DENSITY:
        return 1
    else:
        return 0

def total_energy(g):
        energy = 0
        for row in range(GRID_HEIGHT):
            for col in range(GRID_WIDTH):
                energy += count_neighbors(Coordinate(row,col),g)
        return -1 * energy / 2

"""If a cell is a molecule, count how many of its neightbors is also a molecule:"""
def count_neighbors(coord, g):
        count = 0
        col = coord.x
        row = coord.y
        if g[row][col] == 0:
            return 0
        if row > 0 and g[row-1][col] == 1:
            count += 1
        if col > 0 and g[row][col-1] == 1:
            count += 1
        if row < GRID_HEIGHT - 1 and g[row+1][col] == 1:
            count += 1
        if col < GRID_WIDTH - 1 and g[row][col+1] == 1:
            count += 1
        return count

class GridSimulation:
    def __init__(self, root):
        self.root = root
        self.root.title("Kawasaki dynamics")

        # --- Grid Data Structure ---
        # The same 2D list to hold the state of our simulation
        self.grid = [[populate() for _ in range(GRID_WIDTH)] for _ in range(GRID_HEIGHT)]

        # --- Tkinter Canvas Setup ---
        self.canvas = tk.Canvas(root, width=SCREEN_WIDTH, height=SCREEN_HEIGHT, bg=WHITE)
        self.canvas.pack()

        # A 2D list to hold the ID of each rectangle on the canvas
        self.rectangles = [[None for _ in range(GRID_WIDTH)] for _ in range(GRID_HEIGHT)]

        self.create_grid_rectangles()
        self.draw_grid()

         # --- Controls Frame ---
        # A frame to hold all the sliders and controls neatly
        controls_frame = tk.Frame(root)
        controls_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=5)

        self.temperature = tk.DoubleVar()
        self.temperature.set(TEMP_INIT) # Set the initial value

        # 2. Create the Scale (slider) widget.
        self.slider = tk.Scale(
            controls_frame,
            variable=self.temperature,
            from_=TEMP_MIN,                
            to=TEMP_MAX,                
            orient=tk.HORIZONTAL,     
            label="Temperature", 
            resolution=0.01
        )
        self.slider.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=5)
        
        # Bind mouse click event to a handler function
        self.canvas.bind("<Button-1>", self.handle_mouse_click)

        # --- Simulation Loop ---
        self.running = True
        self.update_simulation() # Start the loop

    """Creates the rectangle objects on the canvas once at the start."""
    def create_grid_rectangles(self):
        for row in range(GRID_HEIGHT):
            for col in range(GRID_WIDTH):
                x1 = col * CELL_SIZE
                y1 = row * CELL_SIZE
                x2 = x1 + CELL_SIZE
                y2 = y1 + CELL_SIZE
                # Create a rectangle and store its ID in our rectangles list
                self.rectangles[row][col] = self.canvas.create_rectangle(
                    x1, y1, x2, y2, fill=WHITE, outline=GRID_LINE_COLOR
                )

    """Updates the color of the rectangles on the canvas based on the grid data."""
    def draw_grid(self):
        for row in range(GRID_HEIGHT):
            for col in range(GRID_WIDTH):
                cell_state = self.grid[row][col]
                rect_id = self.rectangles[row][col]
                color = CELL_COLOR if cell_state == 1 else WHITE
                # Use itemconfig to efficiently change the rectangle's fill color
                self.canvas.itemconfig(rect_id, fill=color)

    """Toggles the state of a cell when clicked."""
    def handle_mouse_click(self, event):
        col = event.x // CELL_SIZE
        row = event.y // CELL_SIZE
        
        if 0 <= row < GRID_HEIGHT and 0 <= col < GRID_WIDTH:
            # Toggle the state (0 to 1, or 1 to 0)
            self.grid[row][col] = 1 - self.grid[row][col]

    """Return a random 0-indexed cell (x : horizontal (col),y: vertical (row)) on the grid"""
    def random_cell(self):
        return Coordinate(random.randint(0,GRID_WIDTH-1), random.randint(0,GRID_HEIGHT-1))

    def kawasaki_dynamics(self):
        temperature = float(self.temperature.get())
        if temperature <= 0: return # Avoid division by zero

        c1 = self.random_cell()
        c2 = self.random_cell()

        # # If only one is a molecule
        if self.grid[c1.y][c1.x] != self.grid[c2.y][c2.x]:

            e0 = -1 * (count_neighbors(c1, self.grid) + count_neighbors(c2, self.grid))
            self.grid[c1.y][c1.x] = not self.grid[c1.y][c1.x]
            self.grid[c2.y][c2.x] = not self.grid[c2.y][c2.x]
            e1 = -1 * (count_neighbors(c1, self.grid) + count_neighbors(c2, self.grid))

            q = math.exp((e0 - e1)/float(self.temperature.get()))
            threshold = q/(1+q)

            if not random.random() <= threshold: # if it doesnt meet threshold, flip it back to original state
                self.grid[c1.y][c1.x] = not self.grid[c1.y][c1.x]
                self.grid[c2.y][c2.x] = not self.grid[c2.y][c2.x]


    """This is the main loop for the simulation."""
    def update_simulation(self):
        if not self.running:
            return
        
        # --- Run multiple simulation steps per frame ---
        steps_per_frame = GRID_WIDTH * GRID_HEIGHT
        for _ in range(steps_per_frame):
            self.kawasaki_dynamics()
        
        # --- Drawing ---
        # We only draw once after all the simulation steps are done.
        self.draw_grid()
        
        # --- Schedule the next update ---
        # 1 ms for minimal delays
        self.root.after(1, self.update_simulation)

def main():
    root = tk.Tk()
    app = GridSimulation(root)
    root.mainloop()

if __name__ == "__main__":
    main()