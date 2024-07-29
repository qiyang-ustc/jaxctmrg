import jax
import jax.numpy as jnp

def logZ(env):
    """
        Contract the following figure to calcuate the density of free energy.
        Output: density of log(Z)
    """
    
    M, C1, C2, C3, C4, T1, T2, T3, T4 = env
    ctr9 = jnp.einsum("ab,fga,bcd,cghi,de,eij,kf,lhk,jl->", C1, T4, T1, M, C2, T2, C4, T3, C3)
                        # C1, T4, T1, M, C2, T2, C4, T3, C3
    ctrh6= jnp.einsum("ab,la,bgd,egl,df,fe", C1, C4, T1, T3, C2, C3)
    
    ctrv6= jnp.einsum("la,ab,egl,bgd,df,fe", C1, C2, T4, T2, C3, C4)
    ctr4 = jnp.trace(C1@C2@C3@C4)
    return jnp.log(jnp.abs(ctr9/ctrh6/ctrv6*ctr4))