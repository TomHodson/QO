# external packages
import numpy as np
from matplotlib import pyplot as plt

# local modules
from .general import vec2


def lattice_dispersion(kx, ky, t = vec2(1,1)): return - 2 * t.x * np.cos(kx) - 2 * t.y * np.cos(ky) 
def parabolic_dispersion(kx, ky, E0 = 0, c = 1): return E0 + c*(kx**2 + ky**2)
def linear_dispersion(kx, ky, E0 = 0, c = 1): return E0 + c*np.sqrt(kx**2 + ky**2)

def parabolic_DOS(E, E0 = 0, c = 1): return np.pi / c * np.ones_like(E) / np.size(E) / 2
def linear_DOS(E, E0 = 0, c = 1): return 2*np.pi / c**2 * (E - E0)

def numerical_DOS(dispersion, N, limit_to_edges = True):
    kx = np.linspace(-np.pi, np.pi, N.x)[:, None]
    ky = np.linspace(-np.pi, np.pi, N.y)[None, :]
    sampled_dispersion = dispersion(kx, ky)
    
    min_on_edges = min(
        np.min(sampled_dispersion[0, :]),
        np.min(sampled_dispersion[:, 0]),
        np.min(sampled_dispersion[-1, :]),
        np.min(sampled_dispersion[:, -1]),
    )
    l, h = np.min(sampled_dispersion), min_on_edges if limit_to_edges else np.max(sampled_dispersion)

    E = np.linspace(l,h,20)
    dE = E[1] - E[0]
    number_of_states_less_than_E = np.array([np.sum(e > sampled_dispersion) for e in E])
    DOS = np.diff(number_of_states_less_than_E) / dE / (N.x * N.y)

    f, axes = plt.subplots(ncols = 3, figsize = (15,5))
    axes[0].pcolormesh(kx[:, 0], ky[0, :], sampled_dispersion)
    axes[0].contour(kx[:, 0], ky[0, :], sampled_dispersion, E, colors = 'k', linestyles = '--')
    axes[0].set(title = "Energy in k space with contours")

    axes[1].plot(E, number_of_states_less_than_E/(N.x * N.y)); axes[1].set(ylabel = "# of states < E")
    axes[1].set(title = "Cumulative Energy distribution")

    axes[2].plot(E[1:], DOS)
    axes[2].set(title = "Density of States")
    
    return E, DOS, f, axes