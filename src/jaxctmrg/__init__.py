# SPDX-FileCopyrightText: 2024-present Qi Yang <qiyang@mail.ustc.edu.cn>
#
# SPDX-License-Identifier: MIT

"""
# Motivation
    We implement CTMRG via the state of art scientific computing framework JAX.
    
# Introduction

## Step 1: Set up a bulk tensor and corresponding CTMRG environments

To utilize CTMRG, you need a bulk tensor with 4-legs. You can use built-in dimer-covering bulk tensor `M` to start. Then, you need to use `initenv` to initialize the environments which will be used later in CTMRG program.
    
```python
chi = 16 # bond dimension for CTMRG
ctmrg_iteration = 20 # Number of iterations for CTMRG

from jaxctmrg import *
M = dimercovering()
env = initenv(M, chi)
```

## Step 2: Regist a **update** function

```python
update = update_kit(M, chi, jit=True)
```
    
## Step 3: Update environments via CTMRG
```python
# Exact result will be 0.2915609040
for iter in range(ctmrg_iteration):
    env = update(env)
    print(f"======iter {iter}:\tlog(Z) = {logZ(env)}=======")
```
"""

__all__ = ["dimercovering", "initenv","update_kit","logZ"]

from jaxctmrg.core.update import update_kit
from jaxctmrg.core.env import initenv
from jaxctmrg.core.measure import logZ
from jaxctmrg.io import *
from jaxctmrg.examples import *