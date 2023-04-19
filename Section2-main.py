#%%
import torch as t
import os

from typing import Union, Optional
from einops import reduce, repeat, rearrange
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import matplotlib as mlp
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from matplotlib.pyplot import matshow

from training import generate_synthetic_data, train, ProjectAndRecover, load_models_section2
from visualization import plot_weights_and_bias, visualize_superposition

SMALL_MODELS_PATHNAME = "./model-weights/section2-small/"
BIG_MODELS_PATHNAME = "./model-weights/section2-big/"
SPARSITIES = [0., .7, .9, .97, .99, .997, .999]
device = 'cpu'

#%% Train one model
input_dim = 20
hidden_dim = 5
importance = t.tensor([.7 ** i for i  in range(input_dim)])

sparsity = 0.99                                     # or any float in [0,1)
data = generate_synthetic_data(input_dim, 100000, sparsity)

batch_size = 128
trainloader = DataLoader(tuple((data)), batch_size= batch_size)

model = ProjectAndRecover(input_dim, hidden_dim, importance).to(device).train()
loss = train(model, trainloader, epochs=20, lr=.001)

#%% and visualize
W = model.weights.data
b = model.bias.data
plot_weights_and_bias(W, b)
visualize_superposition(t.tensor(W))


#%% load saved models
small_models = {}
load_models_section2(small_models)

big_models = {}
load_models_section2(big_models, big = True)

#%% visualize saved models
i = 6 # choose i <= 6
model = small_models[SPARSITIES[i]]
plot_weights_and_bias(model.weights.data, model.bias.data)
visualize_superposition(model.weights, sparsity=SPARSITIES[i])

model = big_models[SPARSITIES[i]] 
plot_weights_and_bias(model.weights.data, model.bias.data)
visualize_superposition(model.weights, sparsity=SPARSITIES[i])

# %% visualize all small models
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
plt.show()

fig = plt.figure(figsize=(8, 28))
for index in range(7):
    ax = plt.subplot(171 + index)
    W = small_models[SPARSITIES[index]].weights
    visualize_superposition(W, SPARSITIES[index], ax)
# %% visualize all large models
fig = plt.figure(figsize=(21.6, 3))
grid = ImageGrid(fig, 111,  
                nrows_ncols=(1, 14),
                axes_pad=0.1
                )
plotpairs =[]
for sparsity in SPARSITIES:
    W,b = big_models[sparsity].weights.data, big_models[sparsity].bias.data
    plotpairs += [W.T @ W, b.reshape((len(b), 1))]
for ax, im in zip(grid, plotpairs):
    ax.set_axis_off()
    ax.imshow(im, origin="upper", vmin= -1, vmax= 1, cmap=mlp.colormaps['PiYG'])
plt.show()

fig = plt.figure(figsize=(8, 28))
for index in range(7):
    ax = plt.subplot(171 + index)
    W = big_models[SPARSITIES[index]].weights
    visualize_superposition(W, SPARSITIES[index], ax)
# %%
