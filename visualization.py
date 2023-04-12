#%%
import torch as t
import os

import matplotlib as mlp
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from matplotlib.pyplot import matshow

SMALL_MODELS_PATHNAME = "./model-weights/section2-small"
BIG_MODELS_PATHNAME = "./model-weights/section2-big"

#%%
sparsities = [0., .7, .9, .97, .99, .997, .999]
small_models = {}

for sparsity in sparsities:
    model_filename = SMALL_MODELS_PATHNAME + str(sparsity)
    if os.path.exists(model_filename):
        small_models[sparsity] = t.load(model_filename)
        small_models[sparsity].eval()
    else:
        raise ImportError

#%%
fig = plt.figure(figsize=(14.4, 2))
grid = ImageGrid(fig, 111,  
                nrows_ncols=(1, 14),
                axes_pad=0.1
                )
plotpairs =[]
for sparsity in sparsities:
    W,b = small_models[sparsity].weights.data, small_models[sparsity].bias.data
    plotpairs += [W.T @ W, b.reshape((len(b), 1))]
for ax, im in zip(grid, plotpairs):
    ax.imshow(im, origin="upper", vmin= -1, vmax= 1, cmap=mlp.colormaps['PiYG'])
    ax.set_label(f'Weight matrix and bias for sparsity {sparsity}')
plt.show()

#%% Heat maps

def plot_weights_and_bias(W, b):
    fig = plt.figure(figsize=(5.5, 5))
    grid = ImageGrid(fig, 111,  
                     nrows_ncols=(1, 2),
                     axes_pad=0.2
                     )
    for ax, im in zip(grid, [W.T @ W , b.reshape((len(b), 1))]):
        ax.imshow(im, origin="upper", vmin= -1, vmax= 1, cmap=mlp.colormaps['PiYG'])
    grid[0].set_title(f'Weight matrix and bias for sparsity {sparsity}')
    plt.show()

#%% Visualization
def superposition_metric(W: t.Tensor) -> list[t.Tensor]:
    """computes the following metrics for representation and superposition for all column vectors
    W: input tensor of shape ( _ , num_features)
        
    output:
    representation: tensor of shape (num_features), the j-th entry is the maximum norms of the column vectors W_j
    superposition: tensor of shape (num_features), the j-th entry is the sum \sum_{i \neq j} (W_i * W_j)^2 over the squared inner product of W_j with all other column vectors W_i 
    """
    num_features = W.shape[1]
    matrix = W.T @ W
    representation = reduce(matrix*matrix, 'i j -> j', 'max') **.5 #t.einsum('ij, ij -> j', matrix, matrix) ** .5
    superposition = t.einsum('ij, kl -> ik', matrix, matrix) ** 2
    mask = t.ones((num_features, num_features)) - t.diag_embed(t.ones(num_features))
    superposition = superposition * mask
    superposition = reduce(superposition, 'i j -> j', 'sum')
    return (representation, superposition)

   
        
def visualize_superposition(W):
    representation, superposition = superposition_metric(model.weights)
    
    fig, ax = plt.subplots()
    features = range(num_features) 
    bars = representation.detach().numpy()
    color_values = superposition.detach().numpy()
    color_values = color_values / 1.1 #color_values.max()
    cmap = mlp.colormaps['cividis_r']
    bar_colors = [cmap(color) for color in color_values]
    
    ax.invert_yaxis()
    ax.barh(features, bars, color=bar_colors)
    ax.set_ylabel('features')
    ax.set_box_aspect(2)

    plt.show()

model = small_models[0.]
visualize_superposition(model.weights)

# %%
fig = plt.figure(figsize=(14.4, 2))
grid = ImageGrid(fig, 111,  
                nrows_ncols=(1, 14),
                axes_pad=0.1
                )
plotpairs =[]
for sparsity in sparsities:
    W,b = small_models[sparsity].weights.data, small_models[sparsity].bias.data
    plotpairs += [W.T @ W, b.reshape((len(b), 1))]
for ax, im in zip(grid, plotpairs):
    ax.imshow(im, origin="upper", vmin= -1, vmax= 1, cmap=mlp.colormaps['PiYG'])
    ax.set_label(f'Weight matrix and bias for sparsity {sparsity}')
plt.show()

#%%

