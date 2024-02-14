import numpy as np
from scipy.sparse.linalg import eigsh
from scipy.sparse import kronsum, diags


# Initialize grid.
N = 200               # number of points
x = np.linspace(-1, 1, N)   # grid
X, Y = np.meshgrid(x, x)   # 3D grid

#print(X[X > 0.5])

dx = np.diff(x)[0]         # size step 


def U(X, Y, a, A):
    """ 
    Potential
    """
    ret = np.ones_like(X) * 1e6
    mask = (a < X**2 + Y**2) & (X**2 + Y**2 < A)
    ret[mask] = 0
    return ret

a = 0.1
A = 0.5
k_max = 10

D = diags([1, -2, 1], [-1, 0, 1], shape=(N, N))
D = -1/(2*dx**2) * kronsum(D, D)
V = diags(U(X, Y, a, A).reshape(N**2), 0)
H = D + V

eigval, eigvec = eigsh(H, k=k_max, which='SM')

psi = lambda n: eigvec.T[n].reshape((N,N))

for n in range(k_max):
    np.save(f'data_schro/eigvec_{n}.npy', psi(n))
