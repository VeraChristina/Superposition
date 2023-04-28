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
from visualization import dimensions_per_feature, feature_dimensionality

MODELS_PATHNAME = "./model-weights/section4/"
device = "cpu"

# %% single training run to look for hyper parameters
# input_dim = 200
# hidden_dim = 15
# importance = t.ones(input_dim)

# sparsity = 0.9
# data = generate_synthetic_data(input_dim, 200000, sparsity)

# batch_size = 512
# trainloader = DataLoader(tuple((data)), batch_size=batch_size)

# model = ProjectAndRecover(input_dim, hidden_dim, importance).to(device).train()
# loss = train(model, trainloader, epochs=10, lr=0.01)
# loss = train(model, trainloader, epochs=6, lr=0.001)
# loss = train(model, trainloader, epochs=2, lr=0.0005)

# # # %% and visualize
# W = model.weights.data
# b = model.bias.data
# plot_weights_and_bias(W, b)
# visualize_superposition(t.tensor(W), sparsity)
# print(W[:10, :10])

# %% Train models
NUM_GRIDPOINTS = 40
GRIDPOINTS = t.logspace(0, 1, NUM_GRIDPOINTS)  # log(1) = 0, log(10) = 1
SPARSITIES = -1 / GRIDPOINTS + 1  # 1/(1-sparsities) ranges from 1 to 100 on log scale

num_features = 200
reduce_to_dim = 15
importance = t.ones(num_features)

size_trainingdata = 200000
batch_size = 512

models = {}

losses = [None for _ in range(NUM_GRIDPOINTS)]
losses_filename = MODELS_PATHNAME + "losses"
if os.path.exists(losses_filename):
    losses = t.load(losses_filename)

for index, sparsity in enumerate(SPARSITIES[:8]):
    model_filename = MODELS_PATHNAME + "model" + str(index)
    if os.path.exists(model_filename):
        print("Index", index, ": Loading model from disk: ", model_filename)
        models[index] = (
            ProjectAndRecover(num_features, reduce_to_dim, importance)
            .to(device)
            .train()
        )
        models[index].load_state_dict(t.load(model_filename))
    else:
        print("Index", index, ": Training model from scratch")
        dataset = generate_synthetic_data(num_features, size_trainingdata, sparsity)
        trainloader = DataLoader(tuple((dataset)), batch_size=batch_size)
        models[index] = (
            ProjectAndRecover(num_features, reduce_to_dim, importance)
            .to(device)
            .train()
        )
        loss = train(models[index], trainloader, epochs=10, lr=0.01)
        loss = train(models[index], trainloader, epochs=6, lr=0.001)
        # loss = train(models[index], trainloader, epochs=2, lr=0.0005)
        losses[index] = loss
        t.save(models[index].state_dict(), model_filename)
        t.save(losses, losses_filename)

# %%
