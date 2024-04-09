import torch as t
import matplotlib as mlp
import matplotlib.pyplot as plt

import os
import sys 

current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)

from src.training import generate_synthetic_data, weighted_MSE


NUM_GRIDPOINTS = 40
SPARSITIES = 1 - t.logspace(0, -2, NUM_GRIDPOINTS)

sparsity_index = 19
sparsity = SPARSITIES[sparsity_index]
data= generate_synthetic_data(2, 100000, sparsity)



#Loss Map to visualize local minima -- bias set to [-.1695, .0304] obtained from last training run
loss_map = t.zeros((100, 100))

def get_loss(w_1: t.Tensor, w_2: t.Tensor) -> t.Tensor:
    W = t.stack([w_1, w_2])
    x = t.einsum("i, b i -> b", W, data)
    x = t.relu(t.einsum("i, b -> b i", W, x) + t.tensor([-0.1695, 0.0304]))
    x = weighted_MSE(x, data, t.tensor([1, 1]))
    return x.sum()

for i in range(100):
    for j in range(100):
        loss_map[i][j] = get_loss(t.tensor((i - 50) / 40), t.tensor((j - 50) / 40))


cut_off = t.clamp(loss_map, 0.015, 0.0185)

fig2, ax = plt.subplots()
ax.imshow(cut_off, cmap="cividis")

ax.set_xticks([10, 49, 90])
ax.set_xticklabels([-1, 0, 1])
ax.set_yticks([10, 49, 90])
ax.set_yticklabels([-1, 0, 1])
ax.set_ylabel(r"$w_1$")
ax.set_xlabel(r"$w_2$")
ax.set_title(r"clipped empirical loss for $W=[w_1, w_2]$")
ax.set_label("rel_importance = 1.0, sparsity = 0.9, fixed bias")
plt.show()