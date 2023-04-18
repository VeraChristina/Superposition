import pytest

import torch as t
from utils import assert_all_equal, allclose
from visualization import superposition_metric

def test_superposition_metric():
    matrix_diag = t.cat([t.diag(t.arange(3)), t.diag(t.tensor([0., 0., 3.]))], dim = 1)
    representation_diag, superposition_diag = superposition_metric(matrix_diag)
    
    with pytest.raises(AssertionError):
        assert_all_equal(representation_diag, t.tensor([0.,1.,2.,0.,0.,3.]))
        allclose(superposition_diag, t.tensor([0.0000, 0.0000, 3.0000, 0.0000, 0.0000, 2.0000])**2)
    
