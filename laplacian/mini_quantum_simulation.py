import numpy as np
import matplotlib.pyplot as plt

def laplacian_3d(psi, h):
    """
    Approximate the Laplacian of a 3D wavefunction using finite differences.
    
    Parameters:
    psi : 3D numpy array
        The wavefunction evaluated on a 3D grid.
    h : float
        The grid spacing.
    
    Returns:
    3D numpy array representing the Laplacian of psi.
    """
    laplacian = np.zeros_like(psi)
    laplacian[1:-1,1:-1,1:-1] = (
        psi[2:,1:-1,1:-1] + psi[:-2,1:-1,1:-1] +
        psi[1:-1,2:,1:-1] + psi[1:-1,:-2,1:-1] +
        psi[1:-1,1:-1,2:] + psi[1:-1,1:-1,:-2] -
        6*psi[1:-1,1:-1,1:-1]
    ) / h**2
    return laplacian

def kinetic_energy_operator(psi, h):
    """
    Compute the kinetic energy operator applied to psi:
    T(psi) = -1/2 * Laplacian(psi)
    """
    laplacian = laplacian_3d(psi, h)
    return -0.5 * laplacian

def main():
    # Grid parameters
    N = 50         # number of grid points per dimension
    L = 10.0       # length of the box
    h = L / (N-1)  # grid spacing

    # Generate coordinate grid
    x = np.linspace(-L/2, L/2, N)
    y = np.linspace(-L/2, L/2, N)
    z = np.linspace(-L/2, L/2, N)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    # Sample wavefunction: 3D Gaussian centered at the origin
    alpha = 1.0
    psi = np.exp(-alpha * (X**2 + Y**2 + Z**2))

    # Apply kinetic energy operator
    T_psi = kinetic_energy_operator(psi, h)

    # Output a slice for visualization (e.g. xy-plane at z=0)
    mid = N // 2
    plt.figure(figsize=(8,4))
    plt.subplot(1,2,1)
    plt.title("Wavefunction (z=0 slice)")
    plt.imshow(psi[:,:,mid], extent=[-L/2,L/2,-L/2,L/2], origin='lower')
    plt.colorbar()

    plt.subplot(1,2,2)
    plt.title("Kinetic Energy Operator (z=0 slice)")
    plt.imshow(T_psi[:,:,mid], extent=[-L/2,L/2,-L/2,L/2], origin='lower')
    plt.colorbar()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
