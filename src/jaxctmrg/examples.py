import numpy as np
import jax.numpy as jnp

__all__ = ["non_normal_AFMTriIsing", "dimercovering"]

def non_normal_AFMTriIsing():
    M = np.zeros((2,2,2,2))
    M[1,1,0,0]=1.0
    M[0,0,1,1]=1.0
    M[0,0,0,0]=1.0
    M[0,1,0,0]=1.0
    M[1,1,1,1]=1.0
    M[1,0,1,1]=1.0
    M = M + 0.0j
    return jnp.array(M)

def dimercovering():
    M = np.zeros((2,2,2,2))
    M[1,0,0,0]=1.0
    M[0,1,0,0]=1.0
    M[0,0,1,0]=1.0
    M[0,0,0,1]=1.0
    M = M + 0.0j
    return jnp.array(M)
