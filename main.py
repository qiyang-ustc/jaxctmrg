import numpy as np
import jax
import jax.numpy as jnp
import h5py

import hydra
from omegaconf import DictConfig, OmegaConf

import logging
log = logging.getLogger(__name__)

jax.config.update("jax_enable_x64", True)
jax.numpy.set_printoptions(precision = 20)

import jaxctmrg

from ftensor import fbulk

# Utils for test or poorman run
def getcfg(config_path="conf", config_name="config"):
    with hydra.initialize(version_base=None, config_path=config_path):
        cfg = hydra.compose(config_name="config")
        print(OmegaConf.to_yaml(cfg))
    return cfg

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:    
    chi = cfg.params.chi
    
    # select model
    # with h5py.File("preconditioned.h5", "r") as file:
    #     # file.create_dataset("V", data=params)
    #     M = file["T"][:]
    #     # file.create_dataset("loss", data=loss(params))

    # M = non_normal_AFMTriIsing() # logZ = 0.3230659669
    M = dimercovering() # logZ = 0.2915609040

    # with h5py.File("data/Nv1FF_tensors.h5", "r") as f:
    #     tensor = f["tensor"][...]
    # M,D = fbulk(tensor), tensor.shape[0]**2 # override D
    # print(M.shape,D)
    
    # cfg = getcfg()
    env = initenv(M, chi)
    
    update = update_kit((Dv, Dh), chi, jit=True, 
        # experimental=True)
        experimental=False)
    
    # leftmove(env)
    for iter in range(cfg.optimize.maxiter):
        env = update(env)
        save_env(f"./data/triising/chi_{chi}_env_{iter}.h5", env)
        log.info(f"======iter {iter}:\tlog(Z) = {Z(env)}=======")

if __name__ == "__main__":
    main()
    # beta =1.0 : -