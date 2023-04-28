import torch as t
from utils import assert_shape_equal

from Section3main import train_multiple, evaluate_multiple


def test_multiple():
    input_dim = 2
    hidden_dim = 1
    rel_importances = t.ones((7))

    models, losses = train_multiple(
        input_dim, hidden_dim, rel_importances, 0, no_printing=True, no_training=True
    )
    representation, superposition = evaluate_multiple(models, t.zeros(7 * 10))
    assert_shape_equal(representation, t.ones((7, 2)))
    assert_shape_equal(superposition, t.ones((7, 2)))
