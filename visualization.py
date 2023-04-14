#%%
import torch as t
import os
from einops import rearrange, reduce, repeat

import matplotlib as mlp
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from matplotlib.pyplot import matshow

from training import ProjectAndRecover, load_saved_models, SPARSITIES

SMALL_MODELS_PATHNAME = "./model-weights/section2-small/"
BIG_MODELS_PATHNAME = "./model-weights/section2-big/"
device = 'cpu'

#%% load trained models
if __name__ == "__main__":
    small_models = {}
    load_saved_models(small_models)

    big_models = {}
    load_saved_models(big_models, big = True)

#%% Heat maps
def plot_weights_and_bias(W, b):
    """Plot heat map of W^T@W and b, where W is matrix of shape (m,n) and b vector of length m"""
    fig = plt.figure(figsize=(3.3, 3))
    grid = ImageGrid(fig, 111,  
                     nrows_ncols=(1, 2),
                     axes_pad=0.2
                     )
    for ax, im in zip(grid, [W.T @ W , b.reshape((len(b), 1))]):
        ax.set_axis_off()
        ax.imshow(im, origin="upper", vmin= -1, vmax= 1, cmap=mlp.colormaps['PiYG'])
    # grid[0].set_title(f'Weight matrix and bias for sparsity {sparsity}')
    plt.show()



#%% Visualization
def superposition_metric(matrix: t.Tensor, new: bool = False) -> list[t.Tensor]:
    """Compute metrics for representation and superposition for all column vectors
    
    W: input tensor of shape ( _ , num_features)
    new: boolean -- indicates which superposition metric to compute
        
    Return: pair (representation, superposition) where
    representation: tensor of shape (num_features) whose the j-th entry is the maximum norms of the column vectors W_j
    superposition: tensor of shape (num_features), 
    the j-th entry is the sum \sum_{i \neq j} (W_i * W_j)^2 over the squared inner product of W_j with all other column vectors W_i 
    If new = True, the sum is normalized wrt the norm of W_i
    """
    num_features = matrix.shape[1]
    representation = reduce(matrix*matrix, 'i j -> j', 'sum') ** .5
    
    superposition = t.einsum('ij, il -> jl', matrix, matrix) ** 2
    if new:
        superposition = superposition / (repeat(representation, 'r -> n r', n=num_features) ** 2 + .0001)
        
    mask = t.ones((num_features, num_features)) - t.diag_embed(t.ones(num_features))
    superposition = superposition * mask 
    superposition = reduce(superposition, 'i j -> j', 'sum')
    return (representation, superposition)

        
def visualize_superposition(W: t.Tensor, new: bool = False):
    """Plot histogram of superposition metric wrt all features
    W: input matrix of shape (hidden_dim, num_features)
    new: boolean that indicates which superposition metric to use
    
    bar length in histogram corresponds to representation of features,
    color of bar corresponds to superposition metric of features
    """    
    num_features = W.shape[1]
    representation, superposition = superposition_metric(W, new)
    
    fig, ax = plt.subplots()
    features = range(num_features) 
    bars = representation.detach().numpy()
    color_values = superposition.detach().numpy()
    color_values = color_values / 1.1 #color_values.max()
    cmap = mlp.colormaps['cividis']
    bar_colors = [cmap(color) for color in color_values]
    
    ax.invert_yaxis()
    ax.barh(features, bars, color=bar_colors)
    ax.set_ylabel('features')
    ax.set_box_aspect(2)
    ax.set_axis_off()

    plt.show()
#%% test_visualize_superposition
# matrix = t.diag_embed(t.ones(5))
# print(matrix)
# print(superposition_metric(matrix))

#%%
if __name__ == '__main__':
    i = 1                                   # choose i <= 6
    model = small_models[SPARSITIES[i]] 
    plot_weights_and_bias(model.weights.data, model.bias.data)
    visualize_superposition(model.weights)
#%%
if __name__ == "__main__":
    fig = plt.figure(figsize=(21.6, 3))
    grid = ImageGrid(fig, 111,  
                    nrows_ncols=(1, 14),
                    axes_pad=0.1
                    )
    plotpairs =[]
    for sparsity in SPARSITIES:
        W,b = small_models[sparsity].weights.data, small_models[sparsity].bias.data
        plotpairs += [W.T @ W, b.reshape((len(b), 1))]
    for ax, im in zip(grid, plotpairs):
        ax.set_axis_off()
        ax.imshow(im, origin="upper", vmin= -1, vmax= 1, cmap=mlp.colormaps['PiYG'])
        #ax.set_label(f'W^t W and b for sparsity {sparsity}')
    plt.show()
