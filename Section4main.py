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
from visualization import dimensions_per_feature

MODELS_PATHNAME = "./model-weights/section4/"
device = "cpu"

# %% Train models
NUM_GRIDPOINTS = 40
GRIDPOINTS = t.logspace(0, 2, NUM_GRIDPOINTS)  # log(1) = 0, log(10) = 2
SPARSITIES = -1 / GRIDPOINTS + 1

num_features = 400
reduce_to_dim = 30
importance = t.ones(num_features)

size_trainingdata = 200000
batch_size = 128
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


# %%
