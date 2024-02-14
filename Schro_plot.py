import numpy as np
import matplotlib.pyplot as plt

# Initialize grid.
N = 200               # number of points
x = np.linspace(-1, 1, N)   # grid
X, Y = np.meshgrid(x, x)   # 3D grid

n = 9
psi = np.load(f'data_schro/eigvec_{n}.npy', allow_pickle='TRUE')



fig = plt.figure(1)
ax  = fig.add_subplot(111, projection='3d')
ax.set_title("2D wave function", fontsize=15)
ax.set_xlabel("x", fontsize=15)
ax.set_ylabel("y", fontsize=15)
ax.plot_surface(X, Y, abs(psi)**2, cmap='plasma')

plt.show()
