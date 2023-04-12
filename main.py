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

from training import generate_synthetic_data, train, ProjectAndRecover
from visualization import plot_weights_and_bias, visualize_superposition

SMALL_MODELS_PATHNAME = "./model-weights/section2-small/"
BIG_MODELS_PATHNAME = "./model-weights/section2-big/"
device = 'cpu'

#%% Train one model
input_dim = 20
hidden_dim = 5
importance = t.tensor([.7 ** i for i  in range(input_dim)])

sparsity = 0.7                                      # or any float in [0,1)
data = generate_synthetic_data(20, 100000, sparsity)

batch_size = 128
trainloader = DataLoader(tuple((data)), batch_size= batch_size)

model = ProjectAndRecover(input_dim, hidden_dim, importance).to(device).train()
model = train(model, trainloader, input_dim, hidden_dim, epochs=20)

#%% and visualize
W = model.weights.data
b = model.bias.data
plot_weights_and_bias(W, b)

#%%
visualize_superposition(t.tensor(W))

#%% load saved models
sparsities = [0., .7, .9, .97, .99, .997, .999]

small_models = {}
input_dim = 20                                                     
hidden_dim = 5
importance_factor = .7
importance = t.tensor([importance_factor ** i for i  in range(input_dim)])        

for sparsity in sparsities:
    model_filename = SMALL_MODELS_PATHNAME + str(sparsity)
    if os.path.exists(model_filename):
        small_models[sparsity] = ProjectAndRecover(input_dim, hidden_dim, importance).to(device).train()
        small_models[sparsity].load_state_dict(t.load(model_filename))
        small_models[sparsity].eval()
    else:
        raise ImportError


big_models = {}
input_dim = 80                                                     
hidden_dim = 20
importance_factor = .9
importance = t.tensor([importance_factor ** i for i  in range(input_dim)])        

for sparsity in sparsities:
    model_filename = BIG_MODELS_PATHNAME + str(sparsity)
    if os.path.exists(model_filename):
        big_models[sparsity] = ProjectAndRecover(input_dim, hidden_dim, importance).to(device).train()
        big_models[sparsity].load_state_dict(t.load(model_filename))
        big_models[sparsity].eval()
    else:
        raise ImportError


#%% visualize saved models
i = 2 # choose i <= 6
model = big_models[sparsities[i]] # or big
plot_weights_and_bias(model.weights.data, model.bias.data)
visualize_superposition(model.weights)

# %%
