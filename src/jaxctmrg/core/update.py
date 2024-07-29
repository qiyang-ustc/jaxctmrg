import jax
import jax.numpy as jnp
from functools import partial
import logging
import h5py

def update_kit(M, chi, repeat=10, eps=1E-14, jit=False, verbose=True, experimental=False):
    """
        Regist a jitted **update** function from given shape and config.
    """
    # Dv, Dh = register_D(D)
    Dv, Dh = M.shape[0:2]
    
    move_h = move_kit(Dh, chi, eps, jit, verbose, experimental) 
    move_v = move_kit(Dv, chi, eps, jit, verbose, experimental) 
    
    move_functions = [move_h["L"], move_h["R"], move_v["U"], move_v["D"]]
    def update(environment):
        for move in move_functions:
            for _ in range(repeat):
                environment = move(environment)
        return environment
    
    return update


def move_kit(Dp, chi, eps=1E-14, jit=True, verbose=True, experimental=False):    
    """
        Regist a jitted **move** function from given shape and config
    """
    
    def vanilla_leftmove(env):
        """
            Implementation of  Fig. SuppMat.1 (b) in https://arxiv.org/pdf/1402.2859
        """
        M, C1, C2, C3, C4, T1, T2, T3, T4 = env
        up = jnp.einsum("ab,iha,bcd,chjk,def,eklm,fg,gmn->lnij", C1, T4, T1, M, T1, M, C2, T2).reshape(Dp*chi,Dp*chi)
        dn = jnp.einsum("dc,cba,hfd,ebfg,ljh,igjk,nl,mkn->imae", C4, T4, T3, M, T3, M, C3, T2).reshape(Dp*chi,Dp*chi)

        # try to normalize up and dn, to make the result of (Ru@Rd.T) scale 1.0
        up = up / jnp.linalg.norm(up)
        dn = dn / jnp.linalg.norm(dn)

        Qu, Ru = jnp.linalg.qr(up)
        Qd, Rd = jnp.linalg.qr(dn)
        Qd = Qu
        Rd = Qd.T.conj() @ dn # dn -> Qu @ (Qd.T @ dn) -> 

        U, S, V = jnp.linalg.svd(Ru @ Rd.T)
        # V^dagger s^{-1} U^\dagger = (Ru @ Rd.T)^{-1} = (Rd.T)^{-1} @ Ru^{-1}
        
        nS = 1.0 # Maybe later modify to S.sum()
        S = S/nS
        sqrtiS = jnp.diag(1.0/(jnp.sqrt(S[:chi]))*(S[:chi]>eps))
        
        Pu = (sqrtiS @     V[:chi,:].conj()) @ Rd
        Pd = (sqrtiS @ (U.T)[:chi,:].conj()) @ Ru
        
        if verbose and not jit:
            # Pu.T = Rd.T @ V.T.conj() @ sqrtiS
            # Pd = sqrtiS @ U.T.conj() @ Ru
            # print(Pd @ Pu.T)
            
            logging.info(f"Minimal Singular Value: {S[chi]}, {jnp.sum(S[:chi]>eps)}")
            # print(Pu.T @ Pd)
            # Thus, after all: Pd @ Pu.T ~ 1.0 
            # print(S[chi:])
            logging.info(f"Truncation: {S[chi:].sum()/S.sum()}")

        Pu, Pd = Pu.reshape(chi,chi,Dp), Pd.reshape(chi,chi,Dp)
        
        C1new = jnp.einsum("ab,bcd,kac->kd", C1, T1, Pu)
        C4new = jnp.einsum("dc,hfd,kcf->hk", C4, T3, Pd)
        T4new = jnp.einsum("iha,uac,chjk,dij->dku", T4, Pd, M, Pu)
                
        # Enviroment cosidered contraction error inspectation
        if verbose and not jit:
            C9old = jnp.einsum("ab,fga,bcd,cghi,de,eij,kf,lhk,jl->", C1, T4, T1, M, C2, T2, C4, T3, C3)
            C6new = jnp.einsum("la,ab,egl,bgd,df,fe", C1new, C2, T4new, T2, C3, C4new)
            logging.info(f"Diff Cont9: {1.0-jnp.abs(C6new/nS/C9old)}")
        
        C1new /= jnp.linalg.norm(C1new)
        C4new /= jnp.linalg.norm(C4new)
        T4new /= jnp.linalg.norm(T4new)

        return M, C1new, C2, C3, C4new, T1, T2, T3, T4new

    # Under development, Ignore it in general!
    def experimental_leftmove(env):
        M, C1, C2, C3, C4, T1, T2, T3, T4 = env
        up = jnp.einsum("ab,iha,bcd,chjk,def,eklm,fg,gmn->lnij", C1, T4, T1, M, T1, M, C2, T2).reshape(Dp*chi,Dp*chi)
        dn = jnp.einsum("dc,cba,hfd,ebfg,ljh,igjk,nl,mkn->imae", C4, T4, T3, M, T3, M, C3, T2).reshape(Dp*chi,Dp*chi)

        # try to normalize up and dn, to make the result of (Ru@Rd.T) scale 1.0
        up = up / jnp.linalg.norm(up)
        dn = dn / jnp.linalg.norm(dn)
        
        Ru = up
        Rd = dn
        Qu = jnp.eye(Dp*chi)
        Qd = jnp.eye(Dp*chi)
        
        # Qu, Ru = jnp.linalg.qr(up)
        # Qd, Rd = jnp.linalg.qr(dn)
        # Qd = Qu
        # Rd = Qd.T.conj() @ dn # dn -> Qu @ (Qd.T @ dn) -> 
        # print(Qd.T @ Qu)

        Me = Ru @ Rd.T
        fac = jnp.abs(jnp.trace(Me))
        Ru, Rd = Ru/jnp.sqrt(fac), Rd/jnp.sqrt(fac)

        # U, S, V = jnp.linalg.svd(Me/fac)
        w, v = jnp.linalg.eig(Me/fac) # Me/fac = V @ diag(w) @ V^{-1}
        U, V = v, jnp.linalg.inv(v)
        S = w
        sqrtiS = jnp.diag(1.0/(jnp.sqrt(S[:chi]))*(jnp.abs(S[:chi])>eps))
        
        # logging.info(f"{S-jnp.abs(w)}")
        # sqrtiS = jnp.diag(1.0/(jnp.sqrt(S[:chi]))*(S[:chi]>eps))
        
        # Pu = (sqrtiS @     V[:chi,:].conj()) @ Rd
        # Pd = (sqrtiS @ (U.T)[:chi,:].conj()) @ Ru
        
        Pu = (sqrtiS @     U.T[:chi,:]) @ Rd
        Pd = (sqrtiS @ V[:chi,:]) @ Ru
        
        if verbose:
            # Pu.T = Rd.T @ V.T.conj() @ sqrtiS
            # Pd = sqrtiS @ U.T.conj() @ Ru
            # print(Pd @ Pu.T)
            
            logging.info(f"Minimal Singular Value: {S[chi]}, {jnp.sum(jnp.abs(S[:chi])>eps)}")
            # print(Pu.T @ Pd)
            # Thus, after all: Pd @ Pu.T ~ 1.0 
            # print(S[chi:])
            logging.info(f"Truncation: {S[chi:].sum()/S.sum()}")

        Pu, Pd = Pu.reshape(chi,chi,Dp), Pd.reshape(chi,chi,Dp)
        
        C1new = jnp.einsum("ab,bcd,kac->kd", C1, T1, Pu)
        C4new = jnp.einsum("dc,hfd,kcf->hk", C4, T3, Pd)
        T4new = jnp.einsum("iha,uac,chjk,dij->dku", T4, Pd, M, Pu)
                
        # Enviroment cosidered contraction error inspectation
        if verbose:
            loss_from_projector = 1.0 - jnp.trace(Qu @ Ru @ Pu.reshape(chi,chi*Dp).T @ Pd.reshape(chi,chi*Dp) @ Rd.T @ Qd.T) / jnp.trace(Qu @ Ru @ Rd.T @ Qd.T)
            logging.info(f"loss from projector in QRRQ: {loss_from_projector}")
            
            # Uncomment this line to make projector to I
            # Pu = jnp.reshape(jnp.eye(Dp*chi), (Dp*chi, chi, Dp))
            # Pd = jnp.reshape(jnp.eye(Dp*chi), (Dp*chi, chi, Dp))
            
            C1tmp = jnp.einsum("ab,bcd,kac->kd", C1, T1, Pu)
            C4tmp = jnp.einsum("dc,hfd,kcf->hk", C4, T3, Pd)
            T4tmp = jnp.einsum("iha,uac,chjk,dij->dku", T4, Pd, M, Pu)
            
            C1tmp2 = jnp.einsum("ab,bcd,kac->kd", C1, T1, Pu)
            C4tmp2 = jnp.einsum("dc,hfd,kcf->hk", C4, T3, jnp.reshape(jnp.eye(Dp*chi), (Dp*chi, chi, Dp)))
            T4tmp2 = jnp.einsum("iha,uac,chjk,dij->dku", T4, Pd, M, jnp.reshape(jnp.eye(Dp*chi), (Dp*chi, chi, Dp)))
            
            C1tmp3 = jnp.einsum("ab,bcd,kac->kd", C1, T1, jnp.reshape(jnp.eye(Dp*chi), (Dp*chi, chi, Dp)))
            C4tmp3 = jnp.einsum("dc,hfd,kcf->hk", C4, T3, Pd)
            T4tmp3 = jnp.einsum("iha,uac,chjk,dij->dku", T4, jnp.reshape(jnp.eye(Dp*chi), (Dp*chi, chi, Dp)), M, Pu)
            
            C9old = jnp.einsum("ab,fga,bcd,cghi,de,eij,kf,lhk,jl->", C1, T4, T1, M, C2, T2, C4, T3, C3)
            C6new = jnp.einsum("la,ab,egl,bgd,df,fe", C1tmp, C2, T4tmp, T2, C3, C4tmp)
            C6new2 = jnp.einsum("la,ab,egl,bgd,df,fe", C1tmp2, C2, T4tmp2, T2, C3, C4tmp2)
            C6new3 = jnp.einsum("la,ab,egl,bgd,df,fe", C1tmp3, C2, T4tmp3, T2, C3, C4tmp3)
            
            logging.info(f"Rd norm, Ru norm: {jnp.linalg.norm(Rd)}, {jnp.linalg.norm(Ru)}")
            logging.info(f"Diff Cont9: {1.0-jnp.abs(C6new/C9old)}")
            logging.info(f"Diff Cont9: {1.0-jnp.abs(C6new2/C9old)}")
            logging.info(f"Diff Cont9: {1.0-jnp.abs(C6new3/C9old)}")

            # entanglement before and after power:
            _, S, _ = jnp.linalg.svd(C4 @ C1)
            p =  S**2 / jnp.sum(S**2)
            p1 = p
            entanglement_before = -jnp.sum(p * jnp.log(p+1E-8))
            _, S, _ = jnp.linalg.svd(C4new @ C1new)
            p =  S**2 / jnp.sum(S**2)
            p2 = p
            entanglement_after = -jnp.sum(p * jnp.log(p+1E-8))
            logging.info(f"Entanglement before and after power: {entanglement_before}, {entanglement_after}")
            print(p1, p2)
            
        C1new /= jnp.linalg.norm(C1new)
        C4new /= jnp.linalg.norm(C4new)
        T4new /= jnp.linalg.norm(T4new)

        return M, C1new, C2, C3, C4new, T1, T2, T3, T4new

    if experimental:
        logging.info(f"Left Move Registed for D={Dp}. Employ experimental Leftmove!")
        leftmove = experimental_leftmove
    else:
        logging.info(f"Left Move Registed for D={Dp}. Employ vanilla Leftmove!")
        leftmove = vanilla_leftmove

    def circle90(env):
        M, C1, C2, C3, C4, T1, T2, T3, T4 = env
        return M.transpose(1,2,3,0), C2, C3, C4, C1, T2, T3, T4, T1

    def circle180(env):
        M, C1, C2, C3, C4, T1, T2, T3, T4 = env
        return M.transpose(2,3,0,1), C3, C4, C1, C2, T3, T4, T1, T2

    def circle270(env):
        M, C1, C2, C3, C4, T1, T2, T3, T4 = env
        return M.transpose(3,0,1,2), C4, C1, C2, C3, T4, T1, T2, T3

    def rightmove(env):
        return circle180(leftmove(circle180(env)))

    def upmove(env):
        return circle90(leftmove(circle270(env)))
    
    def downmove(env):
        return circle270(leftmove(circle90(env)))
    
    moves = leftmove, rightmove, upmove, downmove
    
    if jit:
        moves = list(map(jax.jit,moves))
        
    move_functions={"L":moves[0], "R":moves[1], "U":moves[2], "D":moves[3]}
    return move_functions