import itertools
import numpy as np

def fourier_basis(obs, order):
    #X is assumed to be normalized
    iter = itertools.product(range(order+1), repeat=obs.shape[0])
    multipliers = np.array([list(map(int,x)) for x in iter])

    basisFeatures = np.array([obs[i] for i in range(obs.shape[0])])
    basis = np.cos(np.pi * np.dot(multipliers, basisFeatures))
    return basis