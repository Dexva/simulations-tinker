import tkinter as tk
import time
import numpy as np
import random
import math
import copy

# --- Grid Constants ---
GRID_WIDTH = 15     # Number of cells horizontally
GRID_HEIGHT = 15      # Number of cells vertically
CELL_SIZE = 20        # Size of each cell in pixels
SCREEN_WIDTH = GRID_WIDTH * CELL_SIZE
SCREEN_HEIGHT = GRID_HEIGHT * CELL_SIZE

# --- Simulation Constants ---
density = 0.8

# Colors
WHITE = "#ffffff" 
BLACK = "#000000"
GRID_LINE_COLOR = "#c8c8c8" 
CELL_COLOR = "#34cfeb" 

class Coordinate:
    def __init__(self, x_coord, y_coord):
        self.x = x_coord
        self.y = y_coord

def calculate_energy(g):
        energy = 0
        for row in range(GRID_HEIGHT):
            for col in range(GRID_WIDTH):
                energy += count_neighbors(Coordinate(row,col),g)
        return -1 * energy / 2

def count_neighbors(coord, g):
        """
        If a cell is a molecule, count how many of its neightbors is a molecule:
        """
        count = 0
        x = coord.x
        y = coord.y
        if g[x][y] == 0:
            return 0

        if x!=0 and g[x-1][y] == 1:
            count += 1
        if y!=0 and g[x][y-1] == 1:
            count += 1
        if x!=GRID_WIDTH-1 and g[x+1][y] == 1:
            count += 1
        if y!=GRID_HEIGHT-1 and g[x][y+1] == 1:
            count += 1
        return count
        

class GridSimulation:
    def __init__(self, root):
        self.root = root
        self.root.title("Gas-Liquid Transition (Kawasaki dynamics)")

        # --- Grid Data Structure ---
        # The same 2D list to hold the state of our simulation
        self.grid = [[round(density * random.random()) for _ in range(GRID_WIDTH)] for _ in range(GRID_HEIGHT)]
        
        # --- Tkinter Canvas Setup ---
        self.canvas = tk.Canvas(root, width=SCREEN_WIDTH, height=SCREEN_HEIGHT, bg=WHITE)
        self.canvas.pack()

        # A 2D list to hold the ID of each rectangle on the canvas
        self.rectangles = [[None for _ in range(GRID_WIDTH)] for _ in range(GRID_HEIGHT)]

        self.create_grid_rectangles()
        self.draw_grid()

        # --- SLIDER SETUP ---
        self.temperature = tk.DoubleVar()
        self.temperature.set(0.1) # Set the initial value

        # 2. Create the Scale (slider) widget.
        self.slider = tk.Scale(
            root,
            variable=self.temperature,
            from_=0.1,                
            to=5,                
            orient=tk.HORIZONTAL,     
            label="Temperature", 
            resolution=0.1
        )
        self.slider.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=5)

        # Bind mouse click event to a handler function
        self.canvas.bind("<Button-1>", self.handle_mouse_click)

        # --- Simulation Loop ---
        self.running = True
        self.update_simulation() # Start the loop

    def create_grid_rectangles(self):
        """
        Creates the rectangle objects on the canvas once at the start.
        """
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

    def draw_grid(self):
        """
        Updates the color of the rectangles on the canvas based on the grid data.
        """
        for row in range(GRID_HEIGHT):
            for col in range(GRID_WIDTH):
                cell_state = self.grid[row][col]
                rect_id = self.rectangles[row][col]
                
                color = CELL_COLOR if cell_state == 1 else WHITE
                
                # Use itemconfig to efficiently change the rectangle's fill color
                self.canvas.itemconfig(rect_id, fill=color)

    def handle_mouse_click(self, event):
        """
        Toggles the state of a cell when clicked.
        """
        col = event.x // CELL_SIZE
        row = event.y // CELL_SIZE
        
        if 0 <= row < GRID_HEIGHT and 0 <= col < GRID_WIDTH:
            # Toggle the state (0 to 1, or 1 to 0)
            self.grid[row][col] = 1 - self.grid[row][col]
            # No need to call draw_grid() here, the main loop handles it

    def random_cell(self):
        """
        Return a random 0-indexed cell (x,y) on the grid
        """
        return Coordinate(random.randint(0,GRID_WIDTH-1), random.randint(0,GRID_HEIGHT-1))
        # return {"x": random.randint(0,GRID_WIDTH-1), "y": random.randint(0,GRID_HEIGHT-1)}



    def update_simulation(self):
        """
        This is the main loop for the simulation.
        """
        if not self.running:
            return
        
        # --- "Kawasaki dynamics" simulation Logic (based on 3B1B video)  ---
        
        # Choose 2 random cells
        c1 = self.random_cell()
        c2 = self.random_cell()

        # # If only one is a molecule
        if self.grid[c1.x][c1.y] != self.grid[c2.x][c2.y]:

            e0 = -1 * (count_neighbors(c1, self.grid) + count_neighbors(c2, self.grid))
            self.grid[c1.x][c1.y] = not self.grid[c1.x][c1.y]
            self.grid[c2.x][c2.y] = not self.grid[c2.x][c2.y]
            e1 = -1 * (count_neighbors(c1, self.grid) + count_neighbors(c2, self.grid))

            q = math.exp((e0 - e1)/float(self.temperature.get()))
            threshold = q/(1+q)

            if not random.random() <= threshold: # if it doesnt meet threshold, flip it back to original state
                self.grid[c1.x][c1.y] = not self.grid[c1.x][c1.y]
                self.grid[c2.x][c2.y] = not self.grid[c2.x][c2.y]



        # --- Drawing ---
        self.draw_grid()
        
        # --- Schedule the next update ---
        self.root.after(1, self.update_simulation)

def main():
    root = tk.Tk()
    app = GridSimulation(root)
    root.mainloop()

if __name__ == "__main__":
    main()
