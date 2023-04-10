#%%
import torch as t
from typing import Union


#%%
FloatOrList = Union[float, list]

def force_list(a: FloatOrList, size:int) -> list:
    """Convert a to a list of floats, if it isn't already."""
    if isinstance(a, list):
        if len(a) < size:
            raise ValueError(a)
        return [float(value) for value in a]
    elif isinstance(a, float):
        return [a for _ in range(size)]
    raise ValueError(a)

#%%
def generate_synthetic_data(
    num_features: int, size: int = 10000, sparsity: FloatOrList = 1.0, importance: FloatOrList = 1.0) -> list:
    sparsity = force_list(sparsity, size)
    importance = force_list(importance, size)
    
    