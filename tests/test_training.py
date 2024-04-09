import torch as t
from einops import reduce
from src.utils import allclose, assert_all_equal

from src.training import generate_synthetic_data, weighted_MSE


def compare_data_sparsity(data, target_sparsity):
    """test whether each feature in data is sparsely distributed, i.e. zero with probability target_sparsity"""
    data = (data == t.tensor([0]))
    empiric_sparsity = reduce(data, 'b f -> f', 'sum') / data.shape[0]
    
    allclose(empiric_sparsity, t.ones(data.shape[1]) * target_sparsity, .01)


def test_generate_synthetic_data():
    sparsity = 0.7
    data = generate_synthetic_data(30, 500000, sparsity)
    
    compare_data_sparsity(data, sparsity)


def test_weighted_MSE():
    x = t.ones(5)
    x_hat = t.zeros(5)
    weights = t.arange(5)
    assert weighted_MSE(x, x_hat, weights) == t.tensor(10/5)
    