import taichi as ti
import os
import time

start_time = time.perf_counter()

## - BOUNDARY CONDITIONS -- ##

L = 1.0                         # size of the simulation domain
u_inlet = 0.1                   # slower movement
Re = 10000.0                    # Reynolds number
v = u_inlet * L / Re            # kinematic viscosity
injection_size = 20             # size of dye injection box
k = 1.0e4                       # reaction rate constant   
D = 0.0001                      # diffusion coefficient for dyes 

## - SIMUOLATION PARAMETERS -- ##

N = 512                         # grid resolution
dx = L / N                      # grid cell size
dt = 0.005                      # time step, max 0.00007 for explicit viscosity, 0.005 for implicit viscosity
jacobi_iters = 40               # no. of iterations for pressure solving
viscosity_implem = "implicit"   # "explicit" or "implicit"

## - DATA RECORDING STUFF -- ##

jet_colormap = True             # use jet colormap for fluid speed visualization, otherwise grayscale

ti.init(arch=ti.metal)
recording = False               # for saving frames for a video
recording_min_frame = 0
recording_max_frame = 1000

snapshot = False                # for saving single frames
snapshot_target_frame = 1000

data_collection = False         # for collecting data on product yield
end_collection_frame = 10000
experiment_label = "test_1"

simulation_name = "CM-real_geom-2"
simulation_info = f"dt-{dt}_u-{u_inlet}_Re-{Re}_{viscosity_implem}-v-{v:.3f}-k-{k:.3f}"
output_dir = "3_cm_recordings/" +simulation_name + "-" + simulation_info

substep = 1 if viscosity_implem == "implicit" else 50 # sub-stepping for stability


## - FIELD ALLOCATIONS -- ##

velocity = ti.Vector.field(2, dtype=float, shape=(N, N))
new_velocity = ti.Vector.field(2, dtype=float, shape=(N, N))
u_star = ti.Vector.field(2, dtype=float, shape=(N, N))

velocity_magnitude = ti.field(dtype=float, shape=(N, N)) 

pressure = ti.field(dtype=float, shape=(N, N))
new_pressure = ti.field(dtype=float, shape=(N, N))
divergence = ti.field(dtype=float, shape=(N, N))

dye = ti.Vector.field(3,dtype=float, shape=(N, N))
dye_star = ti.Vector.field(3,dtype=float, shape=(N, N)) 
new_dye = ti.Vector.field(3, dtype=float, shape=(N, N))

color_field = ti.Vector.field(3, dtype=float, shape=(N, N))

yield_flux = ti.field(dtype=float, shape=())
total_flux = ti.field(dtype=float, shape=())

## -- UTILITIES -- ##

# Helper function to sample a field at a given index, 
# clamping the indices to be within bounds.
@ti.func
def sample(q, i, j):
    i = ti.max(0, ti.min(N - 1, i))
    j = ti.max(0, ti.min(N - 1, j))
    return q[i, j]

# Helper function for bilinear interpolation 
# of a field q, at some position (x,y)
@ti.func
def bilerp(q, x, y):    
    i = ti.cast(ti.floor(x), int)
    j = ti.cast(ti.floor(y), int)
    
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

# Clamps the normalized speed and calculates the RGB curves
@ti.func
def jet_colormap(x):
    r = ti.max(0.0, ti.min(1.0, 1.5 - ti.abs(4.0 * x - 3.0)))
    g = ti.max(0.0, ti.min(1.0, 1.5 - ti.abs(4.0 * x - 2.0)))
    b = ti.max(0.0, ti.min(1.0, 1.5 - ti.abs(4.0 * x - 1.0)))

    return ti.Vector([r, g, b])

# Calculates the color to display on the GUI
@ti.kernel
def update_colors():    
    max_disp_speed = u_inlet / 2.0 
    
    for i, j in velocity_magnitude:
        # Calculate base fluid color (Jet colormap)
        val = ti.min(velocity_magnitude[i, j] / max_disp_speed, 1.0)
        base_color = jet_colormap(val)

        # Grey-scale fluid -- uncomment to see jet colormap instead
        if not jet_colormap:
            intensity = 0.50 + (val * 0.5) 
            base_color = ti.Vector([intensity, intensity, intensity])

        # Grab the chemical concentrations (clamped between 0 and 1)
        A_val = ti.min(ti.max(dye[i, j][0], 0.0), 1.0)
        B_val = ti.min(ti.max(dye[i, j][1], 0.0), 1.0)
        C_val = ti.min(ti.max(dye[i, j][2], 0.0), 1.0)
        
        # Define the chemical colors
        color_A = ti.Vector([0.0, 1.0, 1.0]) # Cyan
        color_B = ti.Vector([1.0, 0.0, 1.0]) # Magenta
        color_C = ti.Vector([1.0, 1.0, 0.0]) # Yellow
        
        # Total chemical presence (clamped to 1.0 for blending over the background)
        total_dye = ti.min(A_val + B_val + C_val, 1.0)
        final_color = ti.Vector([0.0, 0.0, 0.0]) # Default to black
        
        if total_dye > 0.0:
            # Normalize the mix of chemicals for accurate color representation
            dye_mix = (color_A * A_val + color_B * B_val + color_C * C_val) / (A_val + B_val + C_val)
            final_color = base_color * (1.0 - total_dye) + dye_mix * total_dye
        else:
            final_color = base_color

        if obstacle(i, j):
            final_color = ti.Vector([0.0, 0.0, 0.0]) # Black for obstacles
        
        # Write the blended result to the screen buffer
        color_field[i, j] = final_color

# Helper functioon for calculating the product yield at the outlet
@ti.kernel
def calculate_yield():
    yield_flux[None] = 0.0
    total_flux[None] = 0.0
    
    # Scan the vertical column just before the outlet
    i = N - 2 
    for j in range(N):
        u_x = velocity[i, j].x
        
        # Only count fluid that is actively exiting (moving right)
        if u_x > 0.0 and not obstacle(i, j):

            # Total concentration in this cell
            c_tot = dye[i, j][0] + dye[i, j][1] + dye[i, j][2]
            
            # Integrate concentration * velocity
            total_flux[None] += c_tot * u_x
            yield_flux[None] += dye[i, j][2] * u_x


## -- SIMULATION KERNELS -- ##

#### for boundaries and obstacles

# Generator for rectangular obstacles
@ti.func
def rectangle(i, j, cx, cy, w, h):
    return (i >= cx - w / 2) and (i <= cx + w / 2) and (j >= cy - h / 2) and (j <= cy + h / 2)

# Generator for tilted rectangular obstacles
@ti.func
def tilted_rectangle(i, j, cx, cy, w, h, angle_degrees):
    # Convert degrees to radians
    angle = angle_degrees * 3.14159265 / 180.0
    
    # Translate grid point to the rectangle's center
    dx = float(i) - cx
    dy = float(j) - cy
    
    # Apply inverse rotation matrix
    c = ti.cos(angle)
    s = ti.sin(angle)
    
    rx = dx * c + dy * s
    ry = -dx * s + dy * c
    
    # Check if the rotated point falls inside the axis-aligned bounds
    return (ti.abs(rx) <= w / 2.0) and (ti.abs(ry) <= h / 2.0)

# Generator for circular obstacles
@ti.func
def circle(i, j, cx, cy, radius):
    return (i - cx)**2 + (j - cy)**2 <= radius**2


# Staggered semicircular obstacles
@ti.func
def geometry_1(i,j):
    is_obs = False
    param_y = 0.04

    if circle(i, j, N * 0.3, N * (0.5-param_y), N * 0.04): is_obs = True
    if circle(i, j, N * 0.38, N * (0.5+param_y), N * 0.04): is_obs = True
    if circle(i, j, N * 0.46, N * (0.5-param_y), N * 0.04): is_obs = True
    if circle(i, j, N * 0.54, N * (0.5+param_y), N * 0.04): is_obs = True
    if circle(i, j, N * 0.62, N * (0.5-param_y), N * 0.04): is_obs = True
    if circle(i, j, N * 0.70, N * (0.5+param_y), N * 0.04): is_obs = True
    if circle(i, j, N * 0.78, N * (0.5-param_y), N * 0.04): is_obs = True
    if circle(i, j, N * 0.86, N * (0.5+param_y), N * 0.04): is_obs = True

    return is_obs

# Staggered tiled rectangular obstacles
@ti.func
def geometry_2(i,j):
    is_obs = False
    tilt = 45.0
    barrier_length = 0.15

    if tilted_rectangle(i, j, N * 0.3, N * 0.44, N * barrier_length, N * 0.03, -1*tilt): is_obs = True
    if tilted_rectangle(i, j, N * 0.35, N * 0.56, N * barrier_length, N * 0.03, tilt): is_obs = True
    if tilted_rectangle(i, j, N * 0.4, N * 0.44, N * barrier_length, N * 0.03, -1*tilt): is_obs = True
    if tilted_rectangle(i, j, N * 0.45, N * 0.56, N * barrier_length, N * 0.03, tilt): is_obs = True
    if tilted_rectangle(i, j, N * 0.5, N * 0.44, N * barrier_length, N * 0.03, -1*tilt): is_obs = True
    if tilted_rectangle(i, j, N * 0.55, N * 0.56, N * barrier_length, N * 0.03, tilt): is_obs = True
    if tilted_rectangle(i, j, N * 0.6, N * 0.44, N * barrier_length, N * 0.03, -1*tilt): is_obs = True
    if tilted_rectangle(i, j, N * 0.65, N * 0.56, N * barrier_length, N * 0.03, tilt): is_obs = True
    if tilted_rectangle(i, j, N * 0.7, N * 0.44, N * barrier_length, N * 0.03, -1*tilt): is_obs = True
    if tilted_rectangle(i, j, N * 0.75, N * 0.56, N * barrier_length, N * 0.03, tilt): is_obs = True
    if tilted_rectangle(i, j, N * 0.8, N * 0.44, N * barrier_length, N * 0.03, -1*tilt): is_obs = True
    if tilted_rectangle(i, j, N * 0.85, N * 0.56, N * barrier_length, N * 0.03, tilt): is_obs = True
    if tilted_rectangle(i, j, N * 0.9, N * 0.44, N * barrier_length, N * 0.03, -1*tilt): is_obs = True
    
    return is_obs

# Parent function for generating obstacles in the simulation 
@ti.func
def obstacle(i, j):
    is_obs = False

    # Default flow pathway)
    if rectangle(i, j, N * 0.05, N * 0.5, N * 0.15, N * 0.3): is_obs = True
    if rectangle(i, j, N * 0.6, N * 0.22, N * 0.8, N * 0.45): is_obs = True
    if rectangle(i, j, N * 0.6, N * 0.78, N * 0.8, N * 0.45): is_obs = True

    # Custom obstacle geometry
    if geometry_2(i,j): is_obs = True

    return is_obs

#### for fluid NS-based transport

# Describe the boundary no-slip conditions, including interactions with obstacles
@ti.kernel
def apply_boundary_conditions_midway():
    # Lip-driven cavity conditions
    for i, j in velocity:
        # Inlet wall (left)
        if i==0:
            new_velocity[i, j] = ti.Vector([u_inlet, 0.0])
        # Solid walls on top, bottom 
        if j==0 or j==N-1:
            new_velocity[i, j] = ti.Vector([0.0, 0.0])
        # Right wall  (outlet)
        if i==N-1:
            new_velocity[i, j] = new_velocity[i-1,j]
        # Solid obstacle(s)
        if obstacle(i, j):
            new_velocity[i, j] = ti.Vector([0.0, 0.0])

# First-order semi-Lagrangian advection of the velocity field
@ti.kernel
def advect():
    for i, j in velocity:
        u = velocity[i, j]

        x = i - u.x * dt / dx
        y = j - u.y * dt / dx

        new_velocity[i, j] = bilerp(velocity, x, y) # corresponds to u*

# Explicit implementation of viscosity (forward Euler)
@ti.kernel
def explicit_viscosity():
    alpha = v * dt / (dx * dx)
    for i, j in velocity:
        u_R = sample(velocity, i + 1, j)
        u_L = sample(velocity, i - 1, j)
        u_T = sample(velocity, i, j + 1)
        u_B = sample(velocity, i, j - 1)

        new_velocity[i, j] = (1 - 4 * alpha) * velocity[i, j] + alpha * (u_R + u_L + u_T + u_B)

# Implicit implementation of viscosity (backward Euler, with Jacobi iterations in step_physics)
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

# Calculate the divergence of the velocity field
@ti.kernel
def calc_divergence():
    for i, j in velocity:
        v_R = sample(new_velocity, i + 1, j).x
        v_L = sample(new_velocity, i - 1, j).x
        v_T = sample(new_velocity, i, j + 1).y
        v_B = sample(new_velocity, i, j - 1).y
        divergence[i, j] = 0.5 * (v_R - v_L + v_T - v_B)

# Solve the pressure Poisson equation with Jacobi iterations
@ti.kernel
def solve_pressure_jacobi():
    for i, j, in pressure:
        p_R = sample(pressure, i + 1, j)
        p_L = sample(pressure, i - 1, j)
        p_T = sample(pressure, i, j + 1)
        p_B = sample(pressure, i, j - 1)

        new_pressure[i, j] = (p_R + p_L + p_T + p_B - divergence[i, j]) / 4.0

# Final projection step to find updated velocity field after pressure correction
@ti.kernel
def subtract_pressure_gradient():
    for i, j in velocity:
        p_R = sample(new_pressure, i + 1, j)
        p_L = sample(new_pressure, i - 1, j)
        p_T = sample(new_pressure, i, j + 1)
        p_B = sample(new_pressure, i, j - 1)

        grad = ti.Vector([p_R - p_L, p_T - p_B]) * 0.5

        velocity[i,j] = new_velocity[i,j] - grad

        # --- MATCHING CUSTOM BOUNDARIES --- # 
        # Inlet wall (left)
        if i==0:
            velocity[i, j] = ti.Vector([u_inlet, 0.0])
        # Solid walls on top, bottom 
        if j==0 or j==N-1:
            velocity[i, j] = ti.Vector([0.0, 0.0])
        # Right wall (outlet)
        if i==N-1:
            velocity[i, j] = velocity[i-1,j]
        # Solid obstacle(s)
        if obstacle(i, j):
            velocity[i, j] = ti.Vector([0.0, 0.0])
        

        velocity_magnitude[i,j] = 0.5 * velocity[i,j].norm()

#### for dye mass transport

# Injecting the dyes into the simulation
@ti.kernel
def add_dye():
    box_A_x = int(N * 0.001)
    box_A_y = int(N * 0.3)

    box_B_x = int(N * 0.001)
    box_B_y = int(N * 0.7)

    for i, j in dye:
        if i >= box_A_x and i <= box_A_x + injection_size and j >= box_A_y and j <= box_A_y + injection_size:
            dye[i, j] = ti.Vector([1.0, 0.0, 0.0]) # Dye A
        if i >= box_B_x and i <= box_B_x + injection_size and j >= box_B_y and j <= box_B_y + injection_size:
            dye[i, j] = ti.Vector([0.0, 1.0, 0.0]) # Dye B

# First step of MacCormack advection for dye
@ti.kernel
def advect_dye_star():
    for i, j in dye:
        u = velocity[i, j] # current velocity

        # track back where the fluid came from
        x = i - u.x * dt / dx
        y = j - u.y * dt / dx

        dye_star[i, j] = bilerp(dye, x, y) # corresponds to C*

# Second step of MacCormack advection for dye
@ti.kernel
def advect_maccormack_dye():
    for i, j in dye:
        u = velocity[i, j]
        
        # Step 2: Trace the guess FORWARD in time (+dt)
        x_fwd = i + u.x * dt / dx
        y_fwd = j + u.y * dt / dx
        dye_rev = bilerp(dye_star, x_fwd, y_fwd)
        
        # Step 3: Calculate the interpolation error and correct it
        error = 0.5 * (dye[i, j] - dye_rev)
        corrected_dye = dye_star[i, j] + error
        
        # Step 4: Build the Limiter
        # Find the integer cell coordinates of the original backward trace
        x_back = i - u.x * dt / dx
        y_back = j - u.y * dt / dx
        idx = ti.cast(ti.floor(x_back), int)
        idy = ti.cast(ti.floor(y_back), int)
        
        # Sample the 4 original neighboring cells
        c00 = sample(dye, idx, idy)
        c10 = sample(dye, idx + 1, idy)
        c01 = sample(dye, idx, idy + 1)
        c11 = sample(dye, idx + 1, idy + 1)
        
        # Find the absolute min and max of that localized 2x2 grid
        min_val = ti.min(c00, ti.min(c10, ti.min(c01, c11)))
        max_val = ti.max(c00, ti.max(c10, ti.max(c01, c11)))
        
        # Clamp the corrected value to strictly stay within the physical bounds
        new_dye[i, j] = ti.max(min_val, ti.min(max_val, corrected_dye))

# Diffusion of the dye 
@ti.kernel
def diffuse_dye():
    alpha = D * dt / (dx * dx)
    
    for i, j in dye:
        d_R = sample(dye, i + 1, j)
        d_L = sample(dye, i - 1, j)
        d_T = sample(dye, i, j + 1)
        d_B = sample(dye, i, j - 1)
        
        new_dye[i, j] = dye[i, j] + alpha * (d_R + d_L + d_T + d_B - 4.0 * dye[i, j])

# Ensure dye doesn't go into obstacles. Boundary conditions with the walls are handled by the velocity field
@ti.kernel
def dye_boundary_conditions():
    for i, j in dye:
        if obstacle(i, j):
            new_dye[i, j] = ti.Vector([0.0, 0.0, 0.0])

# Mock reaction between dyes A & B to form a product C. Assume rate = k[A][B]
@ti.kernel
def react_dye():
    for i, j in dye:
        reaction_rate = k * dye[i, j][0] * dye[i, j][1] * dt #reaction rate over time step dt

        # ensure limiting reagents
        limiting_reactant_amount = ti.min(dye[i, j][0], dye[i, j][1])
        actual_reacted = ti.min(reaction_rate, limiting_reactant_amount)

        # update the dye concentration
        dye[i, j][0] -= actual_reacted
        dye[i, j][1] -= actual_reacted
        dye[i, j][2] += actual_reacted # product formed


#### putting it all together 

# Chorin projection method for fluid simulation, coupled with dye transport and reaction
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

    # -- DYE MASS TRANSPORT AND REACTION -- #
    add_dye()
    advect_dye_star()
    advect_maccormack_dye()
    dye_boundary_conditions()
    dye.copy_from(new_dye)

    diffuse_dye()
    dye.copy_from(new_dye)

    react_dye()

def main():    
    # Create a GUI window
    gui = ti.GUI(f"Taichi-FluidSim/{simulation_info}", res=(N, N))
    print(f"Running simulation: {simulation_name} with parameters: {simulation_info}")

    os.makedirs(output_dir, exist_ok=True) if recording or snapshot or data_collection else None
    frame = 0

    data_filename = f"{output_dir}/{simulation_info}_{experiment_label}.csv"
    if data_collection:
        csv_file = open(data_filename, "w")
        csv_file.write("frame,yield_percent,total_flux,product_flux\n")

    
    # Render Loop
    while gui.running:
        for _ in range(substep): # sub-stepping
            step_physics()

        update_colors()

        # Calculate reaction yield at outlet
        calculate_yield()
        reactor_yield = 0.0
        if total_flux[None] > 0.0:
            reactor_yield = (yield_flux[None] / total_flux[None]) * 100.0

        # Display to GUI 
        gui.set_image(color_field)
        gui.text(f"Product Yield: {reactor_yield:.2f}%", pos=(0.55, 0.95), color=0xFFFFFF, font_size=24)

        if recording:           # record a video of the simulation
            filename = f"{output_dir}/frame_{frame:04d}.png"
            if frame <= recording_min_frame:
                        gui.show()
            if frame > recording_min_frame and frame < recording_max_frame:
                        gui.show(filename)
            if frame >= recording_max_frame:
                 break
        elif snapshot:          # take a screenshot of the simulation at a specific frame
            filename = f"{output_dir}/frame_{frame:04d}.png"
            if frame == snapshot_target_frame:
                gui.show(filename)
            if frame > snapshot_target_frame:
                 break
            gui.show()
        elif data_collection:   # collect data on simulation through a .csv file, no GUI to save compute
            csv_file.write(f"{frame},{reactor_yield:.4f},{total_flux[None]:.6f},{yield_flux[None]:.6f}\n")
            if frame % 10 == 0:  # force save every 10 frames
                csv_file.flush()
            if frame >= end_collection_frame:
                break
        else:
            if frame % 1000 == 0:
                print(f"Frame {frame} completed, current yield: {reactor_yield:.2f}%")
            gui.show()

        
        frame += 1

if __name__ == "__main__":
    main()
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"Simulation ran for {elapsed_time:.2f} seconds.")