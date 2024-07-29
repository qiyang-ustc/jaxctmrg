import matplotlib.pyplot as plt
import numpy as np
import h5py
from jaxctmrg.io import load_env
from jaxctmrg.core.update import move_kit
import scipy.linalg

def prec_eig(A):
    w, v = np.linalg.eig(A) # A = v @ np.diag(w) @ np.linalg.inv(v)
    r, q = scipy.linalg.rq(v)
    rAr = np.linalg.inv(r) @ A @ r
    
    wr, vr = np.linalg.eig(rAr)
    u,s,v = np.linalg.svd(A)
    ur,sr,vr = np.linalg.svd(rAr)
    return np.linalg.inv(r), r

iter = 10
chi = 16

env = load_env(f"./data/triising/chi_{chi}_env_{iter}.h5")
M, C1, C2, C3, C4, T1, T2, T3, T4 = env

D, chi = M.shape[0], C1.shape[0]
move_functions = move_kit(D, chi, jit=False, 
        # experimental=True)
        experimental=False)

Tnode2 = np.einsum("ijk,ajl->ilka", T1, T3).reshape(chi**2,chi**2)
Tnode3 = np.einsum("ijk,jpqm,aql->iplkma", T1, M, T3).reshape(chi**2*D, chi**2 * D)
Tnode4 = np.einsum("ijk,jpqm,qabc,dbe->ipaekmcd",T1, M, M, T3).reshape(chi**2*D**2, chi**2*D**2)

w2, vec2 = np.linalg.eig(Tnode2)
w3, vec3 = np.linalg.eig(Tnode3)
w4, vec4 = np.linalg.eig(Tnode4)

u2,s2,v2 = np.linalg.svd(Tnode2)
u3,s3,v3 = np.linalg.svd(Tnode3)
u4,s4,v4 = np.linalg.svd(Tnode4)

l2 = np.einsum("ab,xa->bx", C1, C4).flatten()
l3 = np.einsum("ab,iha,di->dhb", C1, T4, C4).flatten() 
l4 = np.einsum("fg,def,bcd,ab->geca", C1, T4, T4, C4).flatten() 
    
w2[0],w3[0],w4[0], w3[0]-w2[0], w4[0]-w3[0]
s2[0],s3[0],s4[0]

l2.T @ Tnode2 / l2.T, l3.T @ Tnode3 / l3.T, l4.T @ Tnode4 / l4.T

for i in range(100):
    env = move_functions["left"](env)
    M, C1, C2, C3, C4, T1, T2, T3, T4 = env
    Tnode2 = np.einsum("ijk,ajl->ilka", T1, T3).reshape(chi**2,chi**2)
    l2 = np.einsum("ab,xa->bx", C1, C4).flatten()
    print(l2.T @ Tnode2 / l2.T)
    
ir, r = prec_eig(Tnode3)

s_0 = np.linalg.svd((l3.T).reshape(chi*D,chi))[1]
s_np = np.linalg.svd((l3.T @ Tnode3).reshape(chi*D,chi))[1]
s_p = np.linalg.svd( (l3.T @ ir @ Tnode3 @ r).reshape(chi*D,chi))[1]

s_0 / np.sum(s_0), s_np / np.sum(s_np), s_p / np.sum(s_p)

l3_0 = l3

ts2 = np.linalg.svd(v2[:,0].reshape(chi,chi))[1]     
ts3 = np.linalg.svd(v3[:,0].reshape(chi*D,chi))[1]   
ts4 = np.linalg.svd(v4[:,0].reshape(chi*D,chi*D))[1] 
ts2 / np.sum(ts2), ts3 / np.sum(ts3), ts4 / np.sum(ts4)