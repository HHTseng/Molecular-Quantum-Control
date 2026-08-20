"""Blackbody-radiation population propagators (Supplemental Eqs. S6--S8)."""
from __future__ import annotations
import numpy as np
from scipy.linalg import expm

def rate_generator(rates_i_to_j:np.ndarray)->np.ndarray:
    r=np.asarray(rates_i_to_j,dtype=np.float64)
    if r.ndim!=2 or r.shape[0]!=r.shape[1] or np.min(r)<0:raise ValueError("rates must be nonnegative square matrix")
    r=r.copy();np.fill_diagonal(r,0.0)
    # Columns are sources: G[j,i]=R_{i->j}; G[i,i]=-sum_{j!=i}R_{i->j}.
    g=r.T
    np.fill_diagonal(g,-r.sum(axis=1))
    return g

def propagator(generator:np.ndarray,duration_s:float,*,method="exact",microstep_s:float=1e-6)->np.ndarray:
    g=np.asarray(generator,dtype=np.float64)
    if duration_s<0:raise ValueError("negative duration")
    if method=="exact":p=expm(g*duration_s)
    elif method=="paper_euler":
        count=max(1,int(np.ceil(duration_s/microstep_s)));dt=duration_s/count
        p=np.linalg.matrix_power(np.eye(g.shape[0])+g*dt,count)
    else:raise ValueError("unknown method")
    p=np.clip(p.real,0,None);p/=p.sum(axis=0,keepdims=True);return p.astype(np.float32)

def action_propagators(generator:np.ndarray,durations_ms:np.ndarray,**kwargs)->np.ndarray:
    return np.stack([propagator(generator,float(t)*1e-3,**kwargs) for t in durations_ms])

__all__=["rate_generator","propagator","action_propagators"]
