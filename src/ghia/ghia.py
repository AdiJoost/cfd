import matplotlib.pyplot as plt
from src.ghia.ghiaValues import ghia_x, ghia_y
import numpy as np

def plotGhis(uArray, vArray):
    Ny, Nx = uArray.shape

    # Assuming domain is [0,1]x[0,1]
    x = np.linspace(0, 1, Nx)
    y = np.linspace(0, 1, Ny)

    # Extract centerline slices
    u_center = uArray[:, Nx // 2]  # vertical slice at x = 0.5
    v_center = vArray[Ny // 2, :]  # horizontal slice at y = 0.5

    # === Ghia et al. benchmark data (for Re = 100) ===
    ghia_y = np.array([1.0000, 0.9766, 0.9688, 0.9609, 0.9531, 0.8516, 0.7344, 0.6172,
                       0.5000, 0.4531, 0.2813, 0.1719, 0.1016, 0.0703, 0.0625, 0.0547, 0.0000])
    ghia_u = np.array([1.0000, 0.8412, 0.7887, 0.7372, 0.6872, 0.2315, 0.0033, -0.1364,
                       -0.2058, -0.2109, -0.1566, -0.1015, -0.0643, -0.0478, -0.0419, -0.0372, 0.0000])

    ghia_x = np.array([1.0000, 0.9688, 0.9609, 0.9531, 0.8516, 0.7344, 0.6172, 0.5000,
                       0.4531, 0.2813, 0.1719, 0.1016, 0.0703, 0.0625, 0.0547, 0.0000])
    ghia_v = np.array([0.0000, -0.0591, -0.0739, -0.0886, -0.2453, -0.2245, -0.1691,
                       -0.1031, -0.0886, -0.0570, -0.0340, -0.0180, -0.0117, -0.0094, -0.0078, 0.0000])

    # === Plotting ===
    plt.figure(figsize=(12, 5))

    # Plot u vs y (vertical centerline)
    plt.subplot(1, 2, 1)
    plt.plot(y, u_center, label='Simulation (u)', color='blue')
    plt.plot(ghia_y, ghia_u, 'ro', label='Ghia et al. (u)', markersize=5)
    plt.xlabel('y position (x = 0.5)')
    plt.ylabel('u velocity')
    plt.title('Vertical Centerline (u vs y)')
    plt.legend()
    plt.grid(True)

    # Plot v vs x (horizontal centerline)
    plt.subplot(1, 2, 2)
    plt.plot(x, v_center, label='Simulation (v)', color='green')
    plt.plot(ghia_x, ghia_v, 'ro', label='Ghia et al. (v)', markersize=5)
    plt.xlabel('x position (y = 0.5)')
    plt.ylabel('v velocity')
    plt.title('Horizontal Centerline (v vs x)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()