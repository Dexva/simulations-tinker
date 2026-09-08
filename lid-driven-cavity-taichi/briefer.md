# Fluid simulation with Taichi
## Part 1 - Lid-driven cavity

Here I made a 2D fluid simulator based on the Eulerian method (i.e., fields-based rather than particle-based).

The goal for this mini-task is to simulate a fluid enclosed in a box where one side (a "lid") is moving at a constant non-zero velocity. This is known as the [lip-driven cavity problem](https://www.cfd-online.com/Wiki/Lid-driven_cavity_problem) and is a common benchmarking task for fluid simulations.

This project leverages [Taichi](https://pypi.org/project/taichi/), a Python library for performant code that can run on the GPU.


AI Help Disclaimer:
- Ideation & concepts: Gemini
- Coding: Gemini and Copilot