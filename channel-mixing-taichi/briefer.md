# Fluid simulation with Taichi
## Part 2 - Channel Mixing

This is a continuation of the lid-driven cavity simulation in this repository.

The goal for this task is to simulate the mixing of two reagents (dyes) flowing with a fluid under a laminar regime (small Re). I tested two different mixer geometries as a proof-of-concept of how the simulation works. I also compared the mixing yields when we vary parameters, such as fluid velocity and Reynold's number. 

This project leverages [Taichi](https://pypi.org/project/taichi/), a Python library for performant code that can run on the GPU. The code is an extension of `lid-driven-cavity.py`.

AI Help Disclaimer (same with lid-driven cavity simulation):
- Ideation & concepts: Gemini
- Coding: Gemini and Copilot

### How to Use:
You can run the Python script after properly installing and importing Taichi to your local enironment. 

The adjustable simulation parameters are:
- `u_inlet` (fluid velocity at the inlet)
- `Re` (fluid Reynolds number)
- `injection_size` (size of dye injection box)
- `k` (reaction rate constant)
- `D = 0.0001` (diffusion coefficient for dyes)

You may also toggle the following Booleans to record some data:
- `jet_colormap_bool` (Controls the fluids colors; True = cool-warm colorscheme, False = greyscale)
- `recording` (Save frames in a specified range; Can be later stitched as a video)
- `snapshot` (Takes a specified frame and saves it as an image)
- `data_collection` (Collects the reaction yield and fluxes at the outlet up to a certain end frame number and saves it as a .csv file)

Don't forget to adjust the other variables (e.g., target frame number, filenames) as needed.
