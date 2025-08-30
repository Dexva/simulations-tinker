import tkinter as tk
import time
import numpy as np
import random
import math
from PIL import Image, ImageTk

# --- Grid Constants ---
GRID_WIDTH = 40     # Number of cells horizontally
GRID_HEIGHT = 20    # Number of cells vertically
CELL_SIZE = 10      # Size of each cell in pixels
SCREEN_WIDTH = GRID_WIDTH * CELL_SIZE
SCREEN_HEIGHT = GRID_HEIGHT * CELL_SIZE

# --- Simulation Constants ---
TEMP_INIT = 0.01
TEMP_MAX = 2.0
TEMP_MIN = 0.01
POTENTIAL_INIT = -3.0
POTENTIAL_MAX = -1.0
POTENTIAL_MIN = -3.0

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

def total_energy(g):
        energy = 0
        for row in range(GRID_HEIGHT):
            for col in range(GRID_WIDTH):
                energy += count_neighbors(Coordinate(row,col),g)
        return -1 * energy / 2

def count_neighbors(coord, g):
        """
        If a cell is a molecule, count how many of its neightbors is also a molecule:
        """
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
        
def nonselective_count_neighbors(coord, g):
        """
        Count how many of its neightbors is also a molecule, regardless if the cell is 0 or 1
        """
        count = 0
        col = coord.x
        row = coord.y

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
        self.root.title("Glauber Dynamics")

        # --- Grid Data Structure ---
        # Start with an empty grid
        self.grid = [[0 for _ in range(GRID_WIDTH)] for _ in range(GRID_HEIGHT)]
        self.mol_num = 0

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
        
        # --- 2D SLIDER SETUP ---
        self.setup_2d_slider(controls_frame)

        # Bind mouse click event to a handler function
        self.canvas.bind("<Button-1>", self.handle_mouse_click)

        # --- Simulation Loop ---
        self.running = True
        self.update_simulation() # Start the loop


    """Creates and packs the custom 2D slider widget."""
    def setup_2d_slider(self, parent_frame):
        # Frame to group the 2D slider and its labels
        slider_2d_frame = tk.Frame(parent_frame, padx=10, pady=5)
        slider_2d_frame.pack(expand=True) # Center the frame
        
        tk.Label(slider_2d_frame, text="Potential (Y) & Temperature (X)").pack()

        # Define the variables for the 2D slider
        self.temperature_2d_var = tk.DoubleVar(value=TEMP_INIT) # X-axis
        self.potential_var = tk.DoubleVar(value=POTENTIAL_INIT) # Y-axis

        # Define the visual properties of the 2D slider
        self.slider_2d_width = 150
        self.slider_2d_height = 150
        self.knob_radius = 5

        # Create the canvas for the 2D slider area
        self.slider_2d_canvas = tk.Canvas(
            slider_2d_frame, 
            width=self.slider_2d_width, 
            height=self.slider_2d_height, 
            bg=WHITE, 
            borderwidth=1, 
            relief="solid"
        )
        self.slider_2d_canvas.pack()

        bg_image_pil = Image.open("sampled-phase-diagram.png").resize((self.slider_2d_width+10, self.slider_2d_height+10))
        self.slider_bg_photo = ImageTk.PhotoImage(bg_image_pil)
    
        self.slider_2d_canvas.create_image(0, 0, anchor=tk.NW, image=self.slider_bg_photo)

        # Create the draggable knob
        initial_x, initial_y = self._coords_to_pixels(self.temperature_2d_var.get(), self.potential_var.get())
        self.knob_id = self.slider_2d_canvas.create_oval(
            initial_x - self.knob_radius, initial_y - self.knob_radius,
            initial_x + self.knob_radius, initial_y + self.knob_radius,
            fill=KNOB_COLOR, outline=KNOB_COLOR
        )

        # Bind mouse drag events to the canvas
        self.slider_2d_canvas.bind("<B1-Motion>", self._on_knob_drag)

        # Frame for displaying the values
        value_frame = tk.Frame(slider_2d_frame)
        value_frame.pack()
        
        # Create StringVars for display to allow live formatting
        self.temperature_display_var = tk.StringVar()
        self.potential_display_var = tk.StringVar()
        
        # Set initial display values
        self.temperature_display_var.set(f"{self.temperature_2d_var.get():.2f}")
        self.potential_display_var.set(f"{self.potential_var.get():.2f}")

        tk.Label(value_frame, text="Temperature:").pack(side=tk.LEFT)
        tk.Label(value_frame, textvariable=self.temperature_display_var).pack(side=tk.LEFT)
        tk.Label(value_frame, text="Potential:").pack(side=tk.LEFT)
        tk.Label(value_frame, textvariable=self.potential_display_var).pack(side=tk.LEFT)

    """Converts temperature/potential values to pixel coordinates on the 2D slider canvas."""
    def _coords_to_pixels(self, temperature, potential):
        x = (temperature - TEMP_MIN) / (TEMP_MAX - TEMP_MIN) * self.slider_2d_width
        
        potential_range = POTENTIAL_MAX - POTENTIAL_MIN
        if potential_range == 0: return x, self.slider_2d_height / 2
        
        normalized_potential = (potential - POTENTIAL_MIN) / potential_range
        y = (1.0 - normalized_potential) * self.slider_2d_height
        return x, y

    """Handles the logic when the knob is dragged."""
    def _on_knob_drag(self, event):
        x, y = event.x, event.y

        # Clamp the coordinates to stay within the canvas bounds
        x = max(0, min(x, self.slider_2d_width))
        y = max(0, min(y, self.slider_2d_height))

        # Move the visual knob
        self.slider_2d_canvas.coords(self.knob_id, 
            x - self.knob_radius, y - self.knob_radius,
            x + self.knob_radius, y + self.knob_radius)

        # Convert pixel position back to variable values
        temperature = (x / self.slider_2d_width) * (TEMP_MAX - TEMP_MIN) + TEMP_MIN
        
        normalized_potential = 1.0 - (y / self.slider_2d_height)
        potential_range = POTENTIAL_MAX - POTENTIAL_MIN
        potential = normalized_potential * potential_range + POTENTIAL_MIN

        # Update the Tkinter variables
        self.temperature_2d_var.set(round(temperature, 2))
        self.potential_var.set(round(potential, 2))

        # Update the display variables
        self.temperature_display_var.set(f"{temperature:.2f}")
        self.potential_display_var.set(f"{potential:.2f}")

    """Creates the rectangle objects on the canvas once at the start."""
    def create_grid_rectangles(self):
        for row in range(GRID_HEIGHT):
            for col in range(GRID_WIDTH):
                x1 = col * CELL_SIZE
                y1 = row * CELL_SIZE
                x2 = x1 + CELL_SIZE
                y2 = y1 + CELL_SIZE
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
                self.canvas.itemconfig(rect_id, fill=color)

    """Toggles the state of a cell when clicked."""
    def handle_mouse_click(self, event):
        col = event.x // CELL_SIZE
        row = event.y // CELL_SIZE
        
        if 0 <= row < GRID_HEIGHT and 0 <= col < GRID_WIDTH:
            if self.grid[row][col] == 0:
                self.mol_num += 1
            else:
                self.mol_num -= 1
            self.grid[row][col] = 1 - self.grid[row][col]

    """Return a random 0-indexed cell (x,y) on the grid"""
    def random_cell(self):
        return Coordinate(random.randint(0,GRID_WIDTH-1), random.randint(0,GRID_HEIGHT-1))

    def glauber_dynamics(self):
        temperature = float(self.temperature_2d_var.get())
        if temperature <= 0: return

        potential = float(self.potential_var.get())

        c1 = self.random_cell()  
        c1_state = self.grid[c1.y][c1.x]
        c1_neighbors = nonselective_count_neighbors(c1, self.grid)

        e0 = - (c1_state * c1_neighbors) - (potential * self.mol_num)
        
        new_mol_num = self.mol_num
        new_state = 1 - c1_state # The proposed new state (0->1 or 1->0)

        if new_state == 1:
            new_mol_num += 1
        else:
            new_mol_num -= 1
        
        e1 = - (new_state * c1_neighbors) - (potential * new_mol_num)

        # Use Metropolis acceptance criteria
        delta_e = e1 - e0
        if delta_e < 0:
            # If energy is lower, always accept the change
            self.grid[c1.y][c1.x] = new_state
            self.mol_num = new_mol_num
        else:
            # If energy is higher, accept with a probability
            prob = math.exp(-delta_e / temperature)
            if random.random() <= prob:
                self.grid[c1.y][c1.x] = new_state
                self.mol_num = new_mol_num

    """This is the main loop for the simulation."""
    def update_simulation(self):
        if not self.running:
            return

        # Run multiple steps per frame to speed up the simulation
        steps_per_frame = GRID_WIDTH * GRID_HEIGHT
        for _ in range(steps_per_frame):
            self.glauber_dynamics()
        
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

