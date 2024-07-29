import numpy as np
import jax.numpy as jnp
import h5py

__all__ = ["save_env", "load_env"]
env_name_table = ["M", "C1", "C2", "C3", "C4", "T1", "T2", "T3", "T4"]

def save_env(file,env):
    with h5py.File(file, "w") as f:
        for tensor_name in env_name_table:
            f.create_dataset(tensor_name, data=env[env_name_table.index(tensor_name)])
            
def load_env(file):
    with h5py.File(file, "r") as f:
        env = tuple([f[tensor_name][...] for tensor_name in env_name_table])
    return env