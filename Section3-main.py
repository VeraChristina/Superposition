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
from visualization import plot_weights_and_bias, superposition_metric, vectorized_superposition_metric


MODELS_PATHNAME = "./model-weights/section3/"
device = 'cpu'


#%%
NUM_GRIDPOINTS = 20
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
def train_multiple_and_evaluate(input_dim, hidden_dim, rel_importances: t.tensor, sparsity_ind, best: bool = False) -> list:
    num_importances = rel_importances.shape[0]
    num_models = num_importances * 10
    importance_matrix = t.ones((num_models,input_dim))
    importance_matrix[:, input_dim-1] = repeat(rel_importances, 'i -> (r i)', r = 10)
    
    models = ProjectAndRecover(input_dim, hidden_dim, importance_matrix, multiple=num_models)
    losses = train(models, trainloaders[sparsity_ind], epochs = 8, lr= .001, no_printing=False)
    # losses = train(models, trainloaders[sparsity_ind], epochs = 8, lr=0.0001, no_printing=False)
    losses = rearrange(losses, '(r i) -> r i', r = 10)
    # rearrange weights: shape (multiple, hidden_features, input_features) -> (10, num_importances, hidden_features, input_features)
    weights = rearrange(models.weights, '(r i) h f -> r i h f', r = 10)
    print(weights, losses)
    
    representations, superpositions = vectorized_superposition_metric(weights)
    if best == True:
        min_loss_ind = t.argmin(losses, dim=0, keepdim= False)
        representation = t.zeros((num_importances, input_dim))
        superposition = t.zeros((num_importances, input_dim))
        for i, ind in enumerate(min_loss_ind):
            representation[i, :] = representations[ind, i, :]
            superposition[i, :] = superpositions[ind, i, :]
        return representation, superposition
        
    max_loss_ind = t.argmax(losses, dim=0, keepdim=False)
    for i, ind in enumerate(max_loss_ind):
        representations[ind, i, :] = 0
        superpositions[ind, i, :] = 0
    
    representation = reduce(representations, 'r ... -> ...', 'sum') / 9
    superposition = reduce(superpositions, 'r ... -> ...', 'sum') / 9
    
    return representation, superposition

#%% training run for debugging
input_dim = 2
hidden_dim = 1
importances = t.tensor([1])
sparsity_index = 17

representation, superposition = train_multiple_and_evaluate(input_dim, 
                                                            hidden_dim,
                                                            importances, 
                                                            sparsity_index,
                                                            best=True
                                                            )
print(representation, superposition)
# %%
def get_color_2d(x: t.Tensor, y:t.Tensor):
    assert x.shape == y.shape
    shape = y.shape
    transparency_factors = t.concat([repeat(t.ones(shape),'... -> ... r', r =3) , rearrange(y, '... -> ... 1')], dim=-1)
    cmap = mlp.colormaps['cool']
    colors = cmap(x.detach().numpy())
    colors = colors * transparency_factors.detach().numpy()
    return colors
#%%
models_section3 = {}
input_dim = 2
hidden_dim = 1

representation_list = []
superposition_list = []
for sparsity_index in range(NUM_GRIDPOINTS):
    representation, superposition = train_multiple_and_evaluate(input_dim, 
                                                                hidden_dim,
                                                                IMPORTANCES, 
                                                                sparsity_index
                                                                )
    representation_list.append(representation)
    superposition_list.append(superposition)
#%%
superposition = t.stack(superposition_list, dim=0)
representation = t.stack(representation_list, dim =0)


#%%
colors = get_color_2d(superposition[:,:,1], representation[:,:,1])
fig, ax = plt.subplots()
ax.set_axis_off()
ax.imshow(colors)
plt.show()
#%%
print(representation[19,:,:])
print(superposition[19, :,:])
#%%
print(IMPORTANCES)
print(SPARSITIES)

#%%
t.save(superposition, MODELS_PATHNAME + 'superposition')
t.save(representation, MODELS_PATHNAME + 'representation')
#%% print '2d colormap'
matrix=t.zeros((100,100))
colors = repeat(t.linspace(0,1, 100), 't -> r t', r = 100)
transparencies = repeat(t.linspace(1,0,100), 't -> t r', r = 100)
matrix = get_color_2d(colors, transparencies)

fig, ax = plt.subplots()
ax.set_axis_off
ax.imshow(matrix)
plt.show()


# %%
