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
NUM_GRIDPOINTS = 10
SPARSITIES = 1 - t.logspace(0, -2, NUM_GRIDPOINTS)  # sparsities between 0 and .99 corresponding to densities on log scale from 1 to .01 (log(1) = 0, log(.01) = -2)
IMPORTANCES = t.logspace(-1, 1, NUM_GRIDPOINTS)   # relative importances between .1 and 10 on log scale (log(-1) = .1, log(1)= 10)

data={}
trainloaders={}

for ind, sparsity in enumerate(SPARSITIES):
    data[ind] = generate_synthetic_data(2, 100000, sparsity)
    trainloaders[ind]= DataLoader(data[ind], batch_size=128)


# #%%
# def train_and_evaluate(input_dim, hidden_dim, rel_importance, sparsity_ind) -> list:
#     importance = t.ones(input_dim)
#     importance[input_dim-1] = rel_importance
    
#     models=[None for _ in range(5)]
#     losses=[0 for _ in range(5)]
#     for i in tqdm(range(5)):
#         models[i] = ProjectAndRecover(input_dim, hidden_dim, importance)
#         print(models[i].weights)    
#         losses[i] = train(models[i], trainloaders[sparsity_ind], epochs = 10, no_printing=True)
#         print(models[i].weights, losses[i])
    
#     max_loss_ind = losses.index(max(losses))
#     representation = 0
#     superposition = 0
#     for i in range(5):
#         if i != max_loss_ind:
#             rep_add, sup_add = superposition_metric(models[i].weights)
#             representation += rep_add / 4
#             superposition += sup_add / 4
#     return representation, superposition


#%%
def train_multiple_and_evaluate(input_dim, hidden_dim, rel_importance, sparsity_ind) -> list:
    importance = t.ones(input_dim)
    importance[input_dim-1] = rel_importance
    
    models = ProjectAndRecover(input_dim, hidden_dim, importance, multiple=10)
    losses = train(models, trainloaders[sparsity_ind], epochs = 10, no_printing=False)
    print(models.weights, losses)
    
    max_loss_ind = t.argmax(losses)
    representation = 0
    superposition = 0
    for i in range(10):
        if i != max_loss_ind:
            W = models.weights[i] # weights are of shape (multiple, hidden_features, input_features)
            rep_add, sup_add = superposition_metric(W)
            representation += rep_add / 9
            superposition += sup_add / 9
    return representation, superposition

#%%
models_section3 = {}
input_dim = 2
hidden_dim = 1

sparsity_index=5
rel_importance_index = 0
representation, superposition = train_multiple_and_evaluate(input_dim, 
                                                            hidden_dim,
                                                            IMPORTANCES[rel_importance_index], 
                                                            sparsity_index
                                                            )

print(representation, superposition)

# %%
print(SPARSITIES[5])
# %%
