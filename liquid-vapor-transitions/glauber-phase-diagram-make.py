
import matplotlib.pyplot as plt
import math
import numpy as np

# data from glauber-phase-sampler.py
DATA = [
[0, 255, 254, 253, 247, 226, 221, 193, 190, 191],
[0, 255, 255, 248, 243, 225, 201, 192, 183, 179],
[0, 255, 252, 242, 228, 204, 197, 185, 169, 164],
[0, 254, 250, 236, 221, 196, 177, 150, 149, 153],
[0, 0, 233, 222, 184, 157, 149, 147, 135, 127],
[0, 0, 3, 47, 107, 111, 126, 118, 103, 121],
[0, 0, 2, 11, 49, 76, 81, 112, 110, 113],
[0, 0, 0, 9, 16, 43, 64, 93, 98, 99],
[0, 0, 0, 4, 14, 43, 52, 66, 75, 95],
[0, 0, 0, 4, 9, 18, 41, 55, 69, 75],
]


# --- Constants ---
GRID_SIZE = 10       

def display_colormap(data):

    fig, ax = plt.subplots(figsize=(8, 6))

    im = ax.imshow(data, cmap='Blues', interpolation='nearest')

    cbar = fig.colorbar(im)
    cbar.set_label('Value (0-255)', rotation=270, labelpad=15)

    ax.set_title('Liquid-Vapor Phase Diagram (Glauber)')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.show()

def main():
    print("Displaying colormap with Matplotlib...")
    display_colormap(DATA)
    print("Done.")

if __name__ == "__main__":
    main()
