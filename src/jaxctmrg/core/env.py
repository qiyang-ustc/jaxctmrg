import numpy as np
import jax.numpy as jnp


def initenv(M, chi):
    """
        Initialize enviroments corresponding to M and chi.
    """
    D1, D2 = M.shape[0:2]

    C = np.eye(chi) + 0.0j
    T4 = powerinit_T(M, D2, chi)
    T1 = powerinit_T(M.transpose(1,2,3,0), D1, chi)
    T2 = powerinit_T(M.transpose(2,3,0,1), D2, chi)
    T3 = powerinit_T(M.transpose(3,0,1,2), D1, chi)
    env = M, C, C, C, C, T1, T2, T3, T4
    return tuple(map(jnp.array, env))

def powerinit_T(M, D, chi):
    """
        Using Power to initialize enviroments, default.
    """
    T = np.ones(D).reshape(1,D,1)
    while T.shape[0] < chi:
        T = np.einsum("ijk,cjab->iabkc", T, M).reshape(T.shape[0]*D, D, T.shape[0]*D)
    return T[:chi,:,:chi]

def poor_initenv(M, D, chi):
    """
        Randomly initialize enviroments, not suggest.
    """
    C = np.random.rand(chi,  chi) + 0.0j
    T = np.random.rand(chi,D,chi) + 0.0j
    
    C,T = C/np.linalg.norm(C), T/np.linalg.norm(T)
    env = M, C, C, C, C, T, T, T, T
    return tuple(map(jnp.array, env))

