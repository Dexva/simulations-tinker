import taichi as ti
import os

from torch import sub

## - BOUNDARY CONDITIONS -- ##
L = 1.0 # size of the simulation domain
u_wall = 5.0 # velocity of the moving wall
Re = 400.0 # Reynolds number
v = u_wall * L / Re # kinematic viscosity


## - SIMUOLATION PARAMETERS -- ##
N = 512 # grid resolution
dx = L / N  #  grid cell size
dt = 0.00007   #  time step, max 0.00007 for explicit, 0.005 for implicit
jacobi_iters = 40 # no. of iterations for pressure solving
viscosity_implem = "implicit" # "explicit" or "implicit"


## - DATA RECORDING STUFF -- ##
ti.init(arch=ti.metal)
recording = False
simulation_name = "LDC_}"
simulation_info = f"dt-{dt}_u-{u_wall}_Re-{Re}_{viscosity_implem}-v-{v}-colored"
output_dir = "recordings/" +simulation_name + "-" + simulation_info
recording_min_frame = 500
recording_max_frame = 1000
substep = 50
# substep = 1 if viscosity_implem == "implicit" else 50 # sub-stepping for stability


## - FIELD ALLOCATIONS -- ##
velocity = ti.Vector.field(2, dtype=float, shape=(N, N))
new_velocity = ti.Vector.field(2, dtype=float, shape=(N, N))
u_star = ti.Vector.field(2, dtype=float, shape=(N, N)) # holds the advected state [extra buffer]

velocity_magnitude = ti.field(dtype=float, shape=(N, N)) # Scalar field for rendering the speed

pressure = ti.field(dtype=float, shape=(N, N))
new_pressure = ti.field(dtype=float, shape=(N, N))
divergence = ti.field(dtype=float, shape=(N, N))

color_field = ti.Vector.field(3, dtype=float, shape=(N, N))

## -- UTILITIES -- ##

@ti.func
def sample(q, i, j):
    # Helper function to sample a field at a given index, 
    # clamping the indices to be within bounds.
    i = ti.max(0, ti.min(N - 1, i))
    j = ti.max(0, ti.min(N - 1, j))
    return q[i, j]

@ti.func
def bilerp(q, x, y):
    # Find the bottom-left integer cell
    i = ti.cast(ti.floor(x), int)
    j = ti.cast(ti.floor(y), int)
    
    # Calculate the fractional distances
    s = x - i
    t = y - j
    
    # Sample the 4 surrounding cells
    c00 = sample(q, i, j)
    c10 = sample(q, i + 1, j)
    c01 = sample(q, i, j + 1)
    c11 = sample(q, i + 1, j + 1)
    
    # Compute the weighted average
    return (1 - s) * (1 - t) * c00 + s * (1 - t) * c10 + \
           (1 - s) * t * c01 + s * t * c11

@ti.func
def jet_colormap(x):
    # Clamps the normalized speed and calculates the RGB curves
    r = ti.max(0.0, ti.min(1.0, 1.5 - ti.abs(4.0 * x - 3.0)))
    g = ti.max(0.0, ti.min(1.0, 1.5 - ti.abs(4.0 * x - 2.0)))
    b = ti.max(0.0, ti.min(1.0, 1.5 - ti.abs(4.0 * x - 1.0)))
    return ti.Vector([r, g, b])

@ti.kernel
def update_colors():
    for i, j in velocity_magnitude:
        # Normalize the speed. u_wall is our maximum speed (5.0).
        # We multiply by 2.0 to artificially boost the brightness of the slower vortex.
        val = ti.min(velocity_magnitude[i, j] / u_wall * 2.0, 1.0)
        
        # Write the RGB vector to our color field
        color_field[i, j] = jet_colormap(val)

## -- SIMULATION KERNELS -- ##

@ti.kernel
def apply_boundary_conditions_midway():
    # Lip-driven cavity conditions
    for i, j in velocity:
        # Sliding wall (top)
        if j==N-1:
            new_velocity[i, j] = ti.Vector([u_wall, 0.0])
        # Solid walls on left, bottom, right
        if i==0 or j==0 or i==N-1:
            new_velocity[i, j] = ti.Vector([0.0, 0.0])
        

@ti.kernel
def advect():
    for i, j in velocity:
        u = velocity[i, j] # current velocity

        # first order time accurate approximation only
        # track back where the fluid came from
        x = i - u.x * dt / dx
        y = j - u.y * dt / dx

        # new_velocity[i, j] = sample(velocity, ti.cast(x, int), ti.cast(y, int)) # corresponds to u*
        new_velocity[i, j] = bilerp(velocity, x, y) # corresponds to u*

@ti.kernel
def explicit_viscosity():
    alpha = v * dt / (dx * dx)
    for i, j in velocity:
        u_R = sample(velocity, i + 1, j)
        u_L = sample(velocity, i - 1, j)
        u_T = sample(velocity, i, j + 1)
        u_B = sample(velocity, i, j - 1)

        new_velocity[i, j] = (1 - 4 * alpha) * velocity[i, j] + alpha * (u_R + u_L + u_T + u_B)

@ti.kernel
def implicit_viscosity():
    alpha = v * dt / (dx * dx)
    beta = 1.0 + 4.0 * alpha

    for i, j in velocity:
        u_R = sample(velocity, i + 1, j)
        u_L = sample(velocity, i - 1, j)
        u_T = sample(velocity, i, j + 1)
        u_B = sample(velocity, i, j - 1)

        new_velocity[i, j] = (u_star[i,j] + alpha * (u_R + u_L + u_T + u_B)) / beta

    

@ti.kernel
def calc_divergence():
    for i, j in velocity:
        v_R = sample(new_velocity, i + 1, j).x
        v_L = sample(new_velocity, i - 1, j).x
        v_T = sample(new_velocity, i, j + 1).y
        v_B = sample(new_velocity, i, j - 1).y
        divergence[i, j] = 0.5 * (v_R - v_L + v_T - v_B)

@ti.kernel
def solve_pressure_jacobi():
    for i, j, in pressure:
        p_R = sample(pressure, i + 1, j)
        p_L = sample(pressure, i - 1, j)
        p_T = sample(pressure, i, j + 1)
        p_B = sample(pressure, i, j - 1)

        new_pressure[i, j] = (p_R + p_L + p_T + p_B - divergence[i, j]) / 4.0

@ti.kernel
def subtract_pressure_gradient():
    for i, j in velocity:
        p_R = sample(new_pressure, i + 1, j)
        p_L = sample(new_pressure, i - 1, j)
        p_T = sample(new_pressure, i, j + 1)
        p_B = sample(new_pressure, i, j - 1)

        grad = ti.Vector([p_R - p_L, p_T - p_B]) * 0.5

        velocity[i,j] = new_velocity[i,j] - grad

        # --- MATCHING CUSTOM BOUNDARIES --- # --> equivalent to setting up for next step
        # Sliding wall (top)
        if j==N-1:
            velocity[i, j] = ti.Vector([u_wall, 0.0])
        # Solid walls on left, bottom, right
        if i==0 or j==0 or i==N-1:
            velocity[i, j] = ti.Vector([0.0, 0.0])
        

        velocity_magnitude[i,j] = 0.5 * velocity[i,j].norm()


def step_physics():
    advect()

    # -- IMPLICIT VISCOSITY ROUTE -- #
    if viscosity_implem == "implicit":
        u_star.copy_from(new_velocity) # store the advected state for implicit viscosity
        velocity.copy_from(u_star) # initial guess for implicit viscosity

        for _ in range (jacobi_iters):
            implicit_viscosity()
            velocity.copy_from(new_velocity)

        apply_boundary_conditions_midway() 
        calc_divergence()

        for _ in range(jacobi_iters):
            solve_pressure_jacobi()
            pressure.copy_from(new_pressure)

        subtract_pressure_gradient()
    
    # -- EXPLICIT VISCOSITY ROUTE -- #
    if viscosity_implem == "explicit":
        velocity.copy_from(new_velocity)
        explicit_viscosity()
        apply_boundary_conditions_midway()
        calc_divergence() 

        for _ in range(jacobi_iters):
            solve_pressure_jacobi()
            pressure.copy_from(new_pressure)

        subtract_pressure_gradient() 



def main():    
    # Create a GUI window
    gui = ti.GUI(f"Taichi-FluidSim/{simulation_info}", res=(N, N))
    print(f"Running simulation: {simulation_name} with parameters: {simulation_info}")

    os.makedirs(output_dir, exist_ok=True) if recording else None
    frame = 0
    
    # Render Loop
    while gui.running:
        for _ in range(substep): # sub-stepping
            step_physics()


        update_colors()
        # gui.set_image(velocity_magnitude)
        gui.set_image(color_field)

        if recording:
            filename = f"{output_dir}/frame_{frame:04d}.png"
            if frame <= recording_min_frame:
                        gui.show()
            if frame > recording_min_frame and frame < recording_max_frame:
                        gui.show(filename)
            if frame >= recording_max_frame:
                 break
        else:
            gui.show()

        

        frame += 1

if __name__ == "__main__":
    main()