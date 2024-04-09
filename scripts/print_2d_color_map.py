import torch as t
from einops import rearrange, reduce, repeat

import matplotlib as mlp
import matplotlib.pyplot as plt

import sys
import os

# manually add parent folder to sys.path
current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)


from src.visualization import get_color_2d


matrix = t.zeros((100, 100))
colors = repeat(t.linspace(0, 1, 100), "t -> r t", r=100)
transparencies = repeat(t.linspace(1, 0, 100), "t -> t r", r=100)
matrix = get_color_2d(colors, transparencies)

fig1, ax = plt.subplots()
fig1.set_size_inches(2, 2)
ax.set_xticks([0, 99])
ax.set_xticklabels([0, r"$\geq 1$"])
ax.set_xlabel("superposition")
ax.set_yticks([0, 99])
ax.set_yticklabels([r"$\geq1$", 0])
ax.set_ylabel("representation")
ax.imshow(matrix)
plt.show()