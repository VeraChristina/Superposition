import torch as t
import os

from einops import rearrange, reduce, repeat
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import matplotlib as mlp
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

from src.training import (
    ProjectAndRecover,
    generate_synthetic_data,
    train,
)
from src.visualization import vectorized_superposition_metric, get_color_2d


MODELS_PATHNAME = "./model-weights/section3/"
device = "cpu"


NUM_GRIDPOINTS = 40
SPARSITIES = 1 - t.logspace(0, -2, NUM_GRIDPOINTS)
IMPORTANCES = t.logspace(-1, 1, NUM_GRIDPOINTS)
# sparsities between 0 and .99 corresponding to densities on log scale from 1 to .01 (log(1) = 0, log(.01) = -2)
# relative importances between .1 and 10 on log scale (log(.1) = -1, log(10)= 1)


def train_multiple(
    input_dim,
    hidden_dim,
    rel_importances: t.tensor,
    sparsity_ind,
    num_dupl: int = 10,
    no_printing: bool = False,
    no_training: bool = False,
) -> list:
    """Initializes and trains num_dupl ReLU output models for each value in rel_importances,
    with given input_dim, hidden_dim and sparsity index
    If no_printing = True, epoch losses and training progress bar is not printed
    If no_training = True, model is only initialized and losses set to None

    output: one ProjectAndRecover model containing all trained models, weight shape: (num_importances * num_dupl, hidden_dim, input_dim)
    tensor of losses for all trained models, shape: (num_importances * num_dupl)
    """
    num_importances = rel_importances.shape[0]
    num_models = num_importances * num_dupl
    importance_matrix = t.ones((num_models, input_dim))
    importance_matrix[:, input_dim - 1] = repeat(
        rel_importances, "i -> (r i)", r=num_dupl
    )

    models = ProjectAndRecover(
        input_dim, hidden_dim, importance_matrix, multiple=num_models
    )
    if no_training == False:
        losses = train(
            models,
            trainloaders[sparsity_ind],
            epochs=3,
            lr=0.01,
            no_printing=no_printing,
        )
    else:
        losses = None
    return models, losses


def evaluate_multiple(
    models: ProjectAndRecover, losses: t.Tensor, num_dupl: int = 10, best: bool = False
) -> list:
    """for each rel_importances value, compute superposition metrics for corresponding models and
    average the results of all models except the model with highest loss, if best = False, or
    pick the results for the model with lowest loss, if best = True
    models: ProjectAndRecover model trained with train_multiple

    return: superposition and representation tensors
    """
    # rearrange weights: shape (multiple, hidden_features, input_features) -> (num_dupl, num_importances, hidden_features, input_features)
    weights = rearrange(models.weights, "(r i) h f -> r i h f", r=num_dupl)
    losses = rearrange(losses, "(r i) -> r i", r=num_dupl)
    num_importances = losses.shape[1]
    # print(models.weights, models.bias, losses)

    representations, superpositions = vectorized_superposition_metric(weights)
    if best == True:
        min_loss_ind = t.argmin(losses, dim=0, keepdim=False)
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

    representation = reduce(representations, "r ... -> ...", "sum") / (num_dupl - 1)
    superposition = reduce(superpositions, "r ... -> ...", "sum") / (num_dupl - 1)

    return representation, superposition

if __name__ == "__main__":
    data = {}
    trainloaders = {}

    for ind, sparsity in enumerate(SPARSITIES):
        data[ind] = generate_synthetic_data(2, 100000, sparsity)
        trainloaders[ind] = DataLoader(data[ind], batch_size=128)


# single training run for debugging
# if __name__ == '__main__':
#     input_dim = 2
#     hidden_dim = 1
#     importances = t.tensor([1])
#     sparsity_index = 19

#     models, losses = train_multiple(input_dim, hidden_dim, importances, sparsity_index)

#     representation, superposition = evaluate_multiple(models, losses, best=False)
#     print(representation, superposition)
#     print(models.weights, models.bias, losses)


# Training run for grid
if __name__ == "__main__":
    models_section3 = {}
    losses = {}
    input_dim = 2
    hidden_dim = 1

    for sparsity_ind in tqdm(range(NUM_GRIDPOINTS)):
        models_section3[sparsity_ind], losses[sparsity_ind] = train_multiple(
            input_dim,
            hidden_dim,
            IMPORTANCES,
            sparsity_ind,
            no_printing=True,
        )

# Visualization for grid
if __name__ == "__main__":
    representation_list = []
    superposition_list = []
    representation_best_list = []
    superposition_best_list = []

    for sparsity_ind in range(NUM_GRIDPOINTS):
        representation, superposition = evaluate_multiple(
            models_section3[sparsity_ind], losses[sparsity_ind], best=False
        )
        representation_list.append(representation)
        superposition_list.append(superposition)
        representation, superposition = evaluate_multiple(
            models_section3[sparsity_ind], losses[sparsity_ind], best=True
        )
        representation_best_list.append(representation)
        superposition_best_list.append(superposition)

    superposition = t.stack(superposition_list, dim=0)
    representation = t.stack(representation_list, dim=0)
    superposition_best = t.stack(superposition_best_list, dim=0)
    representation_best = t.stack(representation_best_list, dim=0)

    colors = get_color_2d(superposition[:, :, 1], representation[:, :, 1])
    colors_best = get_color_2d(
        superposition_best[:, :, 1], representation_best[:, :, 1]
    )
    fig1, ax = plt.subplots(1, 2)
    ax[0].set_title("averaged results")
    ax[1].set_title("model with smallest loss")
    ax[0].set_xlabel("relative importance")
    ax[1].set_xlabel("relative importance")
    ax[0].set_xticks([-0.5, 19.5, 39.5])
    ax[1].set_xticks([-0.5, 19.5, 39.5])
    ax[0].set_xticklabels([0.1, 1, 10])
    ax[1].set_xticklabels([0.1, 1, 10])
    ax[0].set_ylabel("1 - sparsity")
    ax[0].set_yticks([-0.5, 19.5, 39.5])
    ax[1].set_yticks([])
    ax[0].set_yticklabels([1, 0.1, 0.01])
    ax[0].imshow(colors)
    ax[1].imshow(colors_best)
    plt.show()