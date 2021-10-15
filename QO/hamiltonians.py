# external packages
import numpy as np
from scipy.linalg import eigvalsh

# local modules
from .general import vec2

def k_space_H_no_field(t, k, **kwargs):
    #the factor of two along the diagonal comes from  adding the upper and lower triangle together
    return - 2 * np.sum(np.array(t) * np.cos(k))
    return - 2 * t.x * np.cos(k.x) - 2 * t.y * np.cos(k.y) 

def k_space_H(t, k, L, **kwargs):
    """
         | -2 t_y cos(k_y - l*phi), -t_x, ...     exp(-ik_xL) |
         | -t_x, .                                            |
     H = | 0   ,                                              |
         | .                                                  |
         | exp(ik_xL)  .....                                  |


    """
    phi = 2*np.pi / L.x

    #the factor of two along the diagonal comes from  adding the upper and lower triangle together
    upper_triangle = - t.y * np.diag(np.cos(k.y - np.arange(L.x)*phi), k = 0) \
        - t.x *   np.diag(np.ones(L.x-1), k =  1) \
        + np.exp(-1j * k.x * L.x) * np.diag([1,], k = L.x-1) \
    
    #make hermitian
    H = upper_triangle + upper_triangle.conj().T
    
    return H

def eigs_over_k(hamiltonian, N, t, **kwargs):
    """
    This version works for hamiltionians of arbitary shape
    """
    k_xs = np.linspace(-np.pi, np.pi, N.x)
    k_ys = np.linspace(-np.pi, np.pi, N.y)

    internal_DOF = np.atleast_1d(hamiltonian(t, k = vec2(0, 0), **kwargs)).shape[0]
    eigs = np.zeros(shape = (N.x,N.y,internal_DOF))
    for i, k_x in enumerate(k_xs):
        for j, k_y in enumerate(k_ys):
            k = vec2(k_x, k_y)
            H = np.atleast_2d(hamiltonian(t, k, **kwargs))
            eigs[i, j] = eigvalsh(H)
            
    return eigs.flatten()

def lattice_dispersion(kx, ky, t = vec2(1,1)): return - 2 * t.x * np.cos(kx) - 2 * t.y * np.cos(ky) 

def FS_area(dispersion, N, E):
    kx = np.linspace(-np.pi, np.pi, N.x)[:, None]
    ky = np.linspace(-np.pi, np.pi, N.y)[None, :]
    sampled_dispersion = dispersion(kx, ky)
    
    number_of_states_less_than_E = np.array([np.sum(e > sampled_dispersion) for e in np.atleast_1d(E)])
    areas = number_of_states_less_than_E / (N.x * N.y)
    
    return areas