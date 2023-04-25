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
from visualization import vectorized_superposition_metric, get_color_2d


MODELS_PATHNAME = "./model-weights/section3/"
device = 'cpu'


#%%
NUM_GRIDPOINTS = 20
SPARSITIES = 1 - t.logspace(0, -2, NUM_GRIDPOINTS)  # sparsities between 0 and .99 corresponding to densities on log scale from 1 to .01 (log(1) = 0, log(.01) = -2)
IMPORTANCES = t.logspace(-1, 1, NUM_GRIDPOINTS)   # relative importances between .1 and 10 on log scale (log(-1) = .1, log(1)= 10)

if __name__ == '__main__':
    data={}
    trainloaders={}

    for ind, sparsity in enumerate(SPARSITIES):
        data[ind] = generate_synthetic_data(2, 100000, sparsity)
        trainloaders[ind]= DataLoader(data[ind], batch_size=128)

#%%
def train_multiple(input_dim, hidden_dim, rel_importances: t.tensor, sparsity_ind: int,  no_printing: bool = True) -> list:
    num_importances = rel_importances.shape[0]
    num_models = num_importances * 10
    importance_matrix = t.ones((num_models,input_dim))
    importance_matrix[:, input_dim-1] = repeat(rel_importances, 'i -> (r i)', r = 10)
    
    models = ProjectAndRecover(input_dim, hidden_dim, importance_matrix, multiple=num_models)
    losses = train(models, trainloaders[sparsity_ind], epochs = 3, lr= .01, no_printing=no_printing)
    return models, losses

def evaluate_multiple(models: ProjectAndRecover, losses: t.Tensor, best: bool = False) -> list:
    # rearrange weights: shape (multiple, hidden_features, input_features) -> (10, num_importances, hidden_features, input_features)
    weights = rearrange(models.weights, '(r i) h f -> r i h f', r = 10)
    losses = rearrange(losses, '(r i) -> r i', r = 10)
    num_importances = losses.shape[1]
    print(weights,losses)
    
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


#%% single training run for debugging
if __name__ == '__main__':
    input_dim = 2
    hidden_dim = 1
    importances = t.tensor([1])
    sparsity_index = 10

    models, losses = train_multiple(input_dim, hidden_dim, importances, sparsity_index)

    representation, superposition = evaluate_multiple(models, losses, best=False)
    print(representation, superposition)

#%% Training run for grid
if __name__ == '__main__':
    models_section3 = {}
    losses = {}
    input_dim = 2
    hidden_dim = 1

    representation_list = []
    superposition_list = []
    for sparsity_ind in tqdm(range(NUM_GRIDPOINTS)):
        models_section3[sparsity_ind], losses[sparsity_ind] = train_multiple(input_dim, 
                                                                             hidden_dim,
                                                                             IMPORTANCES, 
                                                                             sparsity_ind,
                                                                             no_printing = True,
                                                                             )

#%% Visualization for grid
if __name__ == '__main__':
    for sparsity_ind in tqdm(range(NUM_GRIDPOINTS)):
        representation, superposition = evaluate_multiple(models_section3[sparsity_ind], losses[sparsity_ind], best=True)
        representation_list.append(representation)
        superposition_list.append(superposition)

    superposition = t.stack(superposition_list, dim=0)
    representation = t.stack(representation_list, dim =0)

    colors = get_color_2d(superposition[:,:,1], representation[:,:,1])
    fig, ax = plt.subplots()
    ax.set_axis_off()
    ax.imshow(colors)
    plt.show()
#%%
if __name__ == '__main__':
    print(representation[18,:,:])
    print(superposition[18, :,:])

# #%%
# if __name__ == '__main__':
#     t.save(superposition, MODELS_PATHNAME + 'superposition_best')
#     t.save(representation, MODELS_PATHNAME + 'representation_best')
#%% print '2d colormap'
if __name__ == '__main__':
    matrix=t.zeros((100,100))
    colors = repeat(t.linspace(0,1, 100), 't -> r t', r = 100)
    transparencies = repeat(t.linspace(1,0,100), 't -> t r', r = 100)
    matrix = get_color_2d(colors, transparencies)

    fig1, ax = plt.subplots()
    ax.set_axis_off()
    ax.imshow(matrix)
    plt.show()


# %%
loss_map = t.zeros((100,100))

def get_loss(w_1: t.Tensor, w_2: t.Tensor) -> t.Tensor:
    W = t.stack([w_1, w_2])
    x = t.einsum('i, b i -> b', W, data[10])
    x = t.relu(t.einsum('i, b -> b i', W, x))
    x = weighted_MSE(x, data[10], t.tensor([1,1.4]))
    return x.sum()

for i in range(100):
    for j in range(100):
        loss_map[i][j] = get_loss(t.tensor((i-50)/48), t.tensor((j-50)/48))
#%%
cut_off = t.clamp(loss_map, 0.0, .017)

fig2, ax = plt.subplots()
ax.imshow(cut_off, cmap='cividis')
ax.set_ylabel(r'$-1 <=w_1 <=1$')
ax.set_xlabel(r'$-1<=w_2<=1$')
ax.set_title(r'clipped loss for $W=[w_1, w_2]$ with rel_importance=1.4, sparsity_ind =10')
plt.show()
    
# %%