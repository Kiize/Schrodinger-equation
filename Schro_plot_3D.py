import numpy as np
import matplotlib.pyplot as plt
import itertools

# Initialize grid.
N = 100              # number of points
x = np.linspace(-1, 1, N)   # grid
y = np.linspace(-1, 1, N)   # grid y
z = np.linspace(-1, 1, N)   # grid z

a = 0.1
A = 0.5
k_max = 3

n = 9
psi = np.load(f'data_schro_3D/eigvec_{n}.npy', allow_pickle='TRUE')
E = np.load(f'data_schro_3D/eigvals.npy', allow_pickle='TRUE')

psi_square = abs(psi)**2

N_plot = N**2
index = np.random.choice(np.arange(0, N**3, 1), p=psi_square,size=N_plot)

tmp = list(itertools.product(x, y, z))

coord = np.array([tmp[c] for c in index])
coord = coord.T

print(E)
plt.figure(0)
plt.plot(E, marker='.', linestyle='')

fig = plt.figure(1)
ax  = fig.add_subplot(111, projection='3d')
ax.set_title("3D wave", fontsize=15)
ax.set_xlabel("x", fontsize=15)
ax.set_ylabel("y", fontsize=15)
ax.set_zlabel("z", fontsize=15)

img = ax.scatter(coord[0], coord[1], coord[2])

plt.show()