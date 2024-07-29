import numpy as np
import h5py
import jax.numpy as jnp

# n1 n2, number of virtual fermions on each bonds
def swapgate(n1,n2):
    S = np.einsum("ij,kl->ikjl",np.eye(2**n1), np.eye(2**n2))
    for i in range(2**n1):
        for j in range(2**n2):
            if bin(i)[2:].count('1') % 2 == 1 and bin(j)[2:].count('1') % 2 == 1:
                S[i,j,i,j] = -S[i,j,i,j]
    return jnp.array(S)

def fdag(T):
    nu,nl,nf,nd,nr = tuple(map(lambda x: len(bin(x)[3:]), T.shape))
    Tdag = T.conj()
    
    Tdag = jnp.einsum("ulfdr,luij,rdpq->jifqp",Tdag, swapgate(nl,nu), swapgate(nr,nd))
    return Tdag


def fbulk(T):
    nu,nl,nf,nd,nr = tuple(map(lambda x: len(bin(x)[3:]), T.shape))
    Tdag = fdag(T)
    S1 = swapgate(nl,nu)
    S2 = swapgate(nd,nr)
    return	jnp.reshape(jnp.einsum("abcde,fgchi,bfjk,dilm->kagjhlme", T,Tdag,S1,S2),(4**nu,4**nl,4**nd,4**nr))
