# %%
import torch as t
import os

from typing import Union, Optional
from einops import reduce, repeat, rearrange
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import matplotlib as mlp
import matplotlib.pyplot as plt

from training import generate_synthetic_data, train, ProjectAndRecover
from visualization import (
    visualize_superposition,
    plot_weights_and_bias,
)  # , dimensions_per_feature

MODELS_PATHNAME = "./model-weights/section4/"
device = "cpu"

# # %% single training run to look for hyper parameters
# input_dim = 200
# hidden_dim = 15
# importance = t.ones(input_dim)

# sparsity = 0.9
# data = generate_synthetic_data(input_dim, 200000, sparsity)
# # %%
# batch_size = 512
# trainloader = DataLoader(tuple((data)), batch_size=batch_size)

# model = ProjectAndRecover(input_dim, hidden_dim, importance).to(device).train()
# loss = train(model, trainloader, epochs=10, lr=0.01)
# loss = train(model, trainloader, epochs=6, lr=0.001)
# loss = train(model, trainloader, epochs=2, lr=0.0005)

# # %% and visualize
# W = model.weights.data
# b = model.bias.data
# plot_weights_and_bias(W, b)
# visualize_superposition(t.tensor(W), sparsity)
# print(W[:10, :10])

# %% Train models
NUM_GRIDPOINTS = 40
GRIDPOINTS = t.logspace(0, 2, NUM_GRIDPOINTS)  # log(1) = 0, log(10) = 2
SPARSITIES = -1 / GRIDPOINTS + 1

num_features = 400
reduce_to_dim = 30
importance = t.ones(num_features)

size_trainingdata = 200000
batch_size = 512
epochs = 25

datasets = {}
trainloaders = {}
models = {}

for sparsity in SPARSITIES:
    datasets[sparsity] = generate_synthetic_data(
        num_features, size_trainingdata, sparsity
    )
    trainloaders[sparsity] = DataLoader(
        tuple((datasets[sparsity])), batch_size=batch_size
    )
    model_filename = MODELS_PATHNAME + "sparsity" + str(sparsity)
    if os.path.exists(model_filename):
        print("Loading model from disk: ", model_filename)
        models[sparsity] = (
            ProjectAndRecover(num_features, reduce_to_dim, importance)
            .to(device)
            .train()
        )
        models[sparsity].load_state_dict(t.load(model_filename))
    else:
        print("Training model from scratch")
        models[sparsity] = (
            ProjectAndRecover(num_features, reduce_to_dim, importance)
            .to(device)
            .train()
        )
        loss = train(models[sparsity], trainloaders[sparsity], epochs=epochs)
        t.save(models[sparsity].state_dict(), model_filename)
