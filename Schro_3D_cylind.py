import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh
from scipy.sparse import diags
import itertools
import time

start = time.time()

# Initialize grid.
N = 100        # number of points
x = np.linspace(-1, 1, N)   # grid
y = np.linspace(-1, 1, N)   # grid y
z = np.linspace(-1, 1, N)   # grid z

dx = np.diff(x)[0]         # size step 

def U(p, a, A):
    """ 
    Potential
    """
    x, y, z = p
    ret = 1e6
    if (a < x**2 + y**2) and (x**2 + y**2 < A):
        ret = 0

    return ret

# Parameters.
a = 0.1 
A = 0.5
k_max = 3

# Hamiltonian.

D = diags([1, -2, 1], [-1, 0, 1], shape=(N, N))
I = sparse.eye(N)
Dxy = sparse.kron(I, D) + sparse.kron(D, I)
Ixy = sparse.kron(I, I)
D_tot = -1/(2*dx**2) * (sparse.kron(I, Dxy) + sparse.kron(D, Ixy)) 

tmp = list(itertools.product(x, y, z))
tmp_diag = np.array([U(point, a, A) for point in tmp])
V = sparse.diags(tmp_diag, 0)


H = D_tot + V

eigval, eigvec = eigsh(H, k=k_max, which='SM')

psi = lambda n: eigvec.T[n]

for n in range(k_max):
    np.save(f'data_schro_3D/eigvec_{n}.npy', psi(n))

print(f'Elapsed time: {time.time() - start:.2f}')

