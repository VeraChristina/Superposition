#%%
import torch as t
import os

from einops import rearrange, reduce, repeat
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import matplotlib as mlp
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

from training import ProjectAndRecover, Config_PaR, generate_synthetic_data, weighted_MSE, train
from visualization import plot_weights_and_bias, superposition_metric

MODELS_PATHNAME = "./model-weights/section3/"
device = 'cpu'


#%%
SPARSITIES = t.linspace(0., .99, 20)
IMPORTANCES = t.linspace(.1, 10., 20)

data={}
trainloaders={}

for ind, sparsity in enumerate(SPARSITIES):
    data[ind] = generate_synthetic_data(2, 10000, sparsity)
    trainloaders[ind]= DataLoader(data[ind], batch_size=128)



#%%
def train_and_evaluate(input_dim, hidden_dim, rel_importance, sparsity_ind) -> list:
    importance = t.ones(input_dim)
    importance[input_dim-1] = rel_importance
    
    models=[None for _ in range(5)]
    losses=[0 for _ in range(5)]
    for i in tqdm(range(5)):
        models[i] = ProjectAndRecover(input_dim, hidden_dim, importance)
        losses[i] = train(models[i], trainloaders[sparsity_ind], epochs = 20, no_printing=True)
        print(models[i].weights, losses[i])
    
    max_loss_ind = losses.index(max(losses))
    representation = 0
    superposition = 0
    for i in range(5):
        if i != max_loss_ind:
            rep_add, sup_add = superposition_metric(models[i].weights)
            representation += rep_add / 4
            superposition += sup_add / 4
    return representation, superposition


#%%
models_section3 = {}
input_dim = 2
hidden_dim = 1

sparsity_index = 19
rel_importance_index = 0
representation, superposition = train_and_evaluate(input_dim, 
                                                   hidden_dim, 
                                                   IMPORTANCES[rel_importance_index], 
                                                   sparsity_index
                                                   )

print(representation, superposition)

# %%
