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
    dimensions_per_feature,
    feature_dimensionality,
    plot_weights_and_bias,
)

MODELS_PATHNAME = "./model-weights/section4/"
device = "cpu"

# %% single training run to look for hyper parameters
# input_dim = 200
# hidden_dim = 15
# importance = t.ones(input_dim)

# sparsity = 0.25
# data = generate_synthetic_data(input_dim, 200000, sparsity)

# batch_size = 512
# trainloader = DataLoader(tuple((data)), batch_size=batch_size)

# model = ProjectAndRecover(input_dim, hidden_dim, importance).to(device).train()
# loss = train(model, trainloader, epochs=10, lr=0.01)
# loss = train(model, trainloader, epochs=10, lr=0.001)
# loss = train(model, trainloader, epochs=25, lr=0.0002)
# print(dimensions_per_feature(model.weights))


# %% Train and save models / Load models
NUM_GRIDPOINTS = 40
GRIDPOINTS = t.logspace(0, 1, NUM_GRIDPOINTS)  # log(1) = 0, log(10) = 1
SPARSITIES = -1 / GRIDPOINTS + 1  # 1/(1-sparsities) ranges from 1 to 10 on log scale

num_features = 200
reduce_to_dim = 15
importance = t.ones(num_features)

size_trainingdata = 200000
batch_size = 512

models = {}

for index, sparsity in enumerate(SPARSITIES):
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
        loss = train(models[index], trainloader, epochs=10, lr=0.001)
        if index < 20:
            loss = train(models[index], trainloader, epochs=50, lr=0.0002)
        elif index < 30:
            loss = train(models[index], trainloader, epochs=30, lr=0.0002)
        else:
            loss = train(models[index], trainloader, epochs=15, lr=0.0002)

        t.save(models[index].state_dict(), model_filename)
# # %% train more
# index = 2
# dataset = generate_synthetic_data(num_features, size_trainingdata, SPARSITIES[index])
# trainloader = DataLoader(tuple((dataset)), batch_size=512)

# models[index] = ProjectAndRecover(num_features, reduce_to_dim, importance).to(device)
# model_filename = MODELS_PATHNAME + "model" + str(index)
# models[index].load_state_dict(t.load(model_filename))
# loss = train(models[index], trainloader, epochs=20, lr=0.0001)
# t.save(models[index].state_dict(), model_filename)
# print(dimensions_per_feature(models[index].weights))

# %% Visualization
dimensionalities = []

for index in range(NUM_GRIDPOINTS):
    weight_matrix = models[index].weights
    dimensionalities.append(dimensions_per_feature(weight_matrix))

fig, ax = plt.subplots()
ax.plot(GRIDPOINTS, dimensionalities)
ax.set_xscale("log")
ax.set_yticks([0, 0.5, 1])
ax.set_xlabel("1/(1-sparsity)")
ax.set_ylabel("dimensions per feature")
plt.show()

# %%
plot_weights_and_bias(models[25].weights.detach(), models[25].bias.detach())
# %%
