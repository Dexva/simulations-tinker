import numpy as np
import random
import math

GRID_WIDTH = 40     
GRID_HEIGHT = 20    
TEMP_INIT = 0.01
TEMP_MAX = 2.0
TEMP_MIN = 0.01
POTENTIAL_INIT = -3.0
POTENTIAL_MAX = -1.0
POTENTIAL_MIN = -3.0
ITER_PER_SAMPLE = GRID_WIDTH * GRID_HEIGHT * 800 # arbitrarily chosen to be big enough to appraoch steady-state
DIAGRAM_RESOLUTION = 10 # no. of ticks in the phase diagram

class Coordinate:
    def __init__(self, x_coord, y_coord):
        self.x = x_coord
        self.y = y_coord

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

def random_cell():
    return Coordinate(random.randint(0,GRID_WIDTH-1), random.randint(0,GRID_HEIGHT-1))

def glauber_dynamics(temperature, potential, grid, mol_num):
    if temperature <= 0: return

    c1 = random_cell()  
    c1_state = grid[c1.y][c1.x]
    c1_neighbors = nonselective_count_neighbors(c1, grid)

    e0 = - (c1_state * c1_neighbors) - (potential * mol_num)
    
    new_mol_num = mol_num
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
        grid[c1.y][c1.x] = new_state
        mol_num = new_mol_num
    else:
        # If energy is higher, accept with a probability
        prob = math.exp(-delta_e / temperature)
        if random.random() <= prob:
            grid[c1.y][c1.x] = new_state
            mol_num = new_mol_num
    
    return grid, mol_num

def sample_simulate(temp, poten):
    grid = [[0 for _ in range(GRID_WIDTH)] for _ in range(GRID_HEIGHT)]
    mol_num = 0

    for _ in range(ITER_PER_SAMPLE):
        grid, mol_num = glauber_dynamics(temp, poten, grid, mol_num)

    # temporary print
    return round(np.mean(np.array(grid)) * 255)

def main():
    temp_step = (TEMP_MAX - TEMP_MIN)/DIAGRAM_RESOLUTION
    poten_step = (POTENTIAL_MAX - POTENTIAL_MIN)/DIAGRAM_RESOLUTION
    
    for p in range (0,DIAGRAM_RESOLUTION):
        row = []
        for t in range(0,DIAGRAM_RESOLUTION):
            curr_temp = TEMP_MIN + t * temp_step
            curr_potential = POTENTIAL_MAX - p * poten_step
            sample_mean = sample_simulate(curr_temp,curr_potential)
            row.append(sample_mean)
        print(row)


if __name__ == "__main__":
    main()