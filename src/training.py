import torch as t
import os

from einops import rearrange, reduce, repeat
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from typing import Optional, Union

SMALL_MODELS_PATHNAME = "./model-weights/section2-small/"
BIG_MODELS_PATHNAME = "./model-weights/section2-big/"
SPARSITIES = [0.0, 0.7, 0.9, 0.97, 0.99, 0.997, 0.999]
device = "cpu"


def generate_synthetic_data(
    num_features: int, size: int = 100000, sparsity: float = 1.0
) -> t.Tensor:
    """generates synthetic data with given feature sparsity and importance
    input:
    num_features    number of features (i.e. the dimension of the synthetic data)
    size            number of data points to be generated
    sparsity        float in [0,1] that indicates how sparsely features are distributed


    output: tensor A of shape (size, num_features),
            each column is a synthetic data point, i.e. a vector of length num_features
            each feature is zero with probability (1-sparsity) and otherwise uniformly distributed in [0, 1]
    """
    assert 0 <= sparsity <= 1
    mask = t.rand(tuple((size, num_features))) < 1 - sparsity
    A = t.rand(size=(size, num_features)) * mask
    return A


class ProjectAndRecover(t.nn.Module):
    """Model architechture according to Section 2 of the paper with option for multiple models

    weights: weight matrix of shape (hidden_features, input_features)
    bias: vector of length input_features
    importance: vector of length input_features, used as weights in the loss function
    multiple: None, in default case of one model, or integer number of models
    """

    def __init__(
        self,
        input_features: int,
        hidden_features: int,
        importance: t.Tensor,
        multiple: Optional[int] = None,
    ):
        """Initialize bias to zero,
        Initialize weights uniformly random in [0,0.8] if multiple == None, uniformly random in [-.5,-.5] otherwise (for Section 3)
        """
        super().__init__()
        if multiple is None:
            weights_shape = (hidden_features, input_features)
            bias_shape = input_features
            self.multiple = False
            assert importance.shape == t.Size([input_features])
            self.weights = t.nn.Parameter(
                t.rand(weights_shape, requires_grad=True) * 0.8
            )
        else:
            weights_shape = (multiple, hidden_features, input_features)
            bias_shape = (multiple, input_features)
            self.multiple = True
            assert importance.shape == t.Size([multiple, input_features])
            self.weights = t.nn.Parameter(
                t.rand(weights_shape, requires_grad=True) - 0.5
            )
        self.bias = t.nn.Parameter(t.zeros(bias_shape, requires_grad=True))
        self.relu = t.nn.ReLU()
        self.importance = importance

    def forward(self, x: t.Tensor) -> t.Tensor:
        """Reduce via linear transformation, recover via its transpose, add bias and apply ReLU

        x: shape (batch, input_features)
        Return: shape (batch, input_features) if multiple == False, shape (batch, 10, input_features) otherwise
        """
        if self.multiple is False:
            x = t.einsum("h i, b i -> b h", self.weights, x)
            x = t.einsum("h i,  b h -> b i", self.weights, x) + self.bias
        else:
            x = t.einsum("m h i, b i -> b m h", self.weights, x)
            x = t.einsum("m h i,  b m h -> b m i", self.weights, x) + self.bias
        return self.relu(x)


class Config_PaR:
    """Configuration for the ProjectAndRecover model with two predefined configurations
    The default configuration is that of small models of Section 2 of the paper
    If big = True, the configuration of big models of Section 2 is initialized
    """

    def __init__(
        self,
        big: bool = False,
        input_dim=20,
        hidden_dim=5,
        importance=t.tensor([0.7**i for i in range(20)]),
        multiple: Optional[int] = None,
    ):
        """If big = True, initialize configuration of big models of Section 2,
        else the default or custom configuration
        """
        if big:
            self.input_dim = 80
            self.hidden_dim = 20
            self.importance = t.tensor([0.9**i for i in range(self.input_dim)])
            self.multiple = None
        else:
            self.input_dim = input_dim
            self.hidden_dim = hidden_dim
            assert importance.shape == (input_dim)
            self.importance = importance
            self.multiple = multiple


def weighted_MSE(x, x_hat, weights: t.Tensor, multiple: bool = False) -> t.Tensor:
    """Compute weighted MSE of x and x^hat wrt weights, with option for multiple models
    x shape: (batch, input_dim)
    x_hat shape: (batch, input_dim) if multiple = False, otherwise shape (batch, num_models, input_dim)

    return tensor of shape: (num_models) if multiple is true, and (1) else
    """
    if multiple == True:
        x_hat = rearrange(x_hat, "b m i -> m b i")
    assert x.shape == x_hat.shape[-2:]
    assert x.shape[-1] == weights.shape[-1]
    squared_error = (x - x_hat) ** 2
    if multiple == True:
        squared_error = rearrange(squared_error, "m b i -> b m i")
    weighted_squared_error = weights * squared_error
    if multiple == False:
        return weighted_squared_error.mean()
    return reduce(weighted_squared_error, "b m i -> m", "mean")


def train(
    model: ProjectAndRecover,
    trainloader: DataLoader,
    epochs: int = 15,
    lr: float = 0.001,
    no_printing=False,
) -> float:
    """Train given model of class ProjectAndRecover on data provided in trainloader,
    return: loss after training

    epochs: number of epochs to train
    lr: learning rate
    """
    optimizer = t.optim.Adam(model.parameters(), lr=lr)
    for epoch in tqdm(range(epochs), disable=no_printing):
        epoch_losses = 0
        epoch_loss = 0
        for i, x in enumerate(trainloader):
            x = x.to(device)
            optimizer.zero_grad()
            x_hat = model(x)
            losses = weighted_MSE(x, x_hat, model.importance, model.multiple)
            loss = losses.sum()
            # can train multiple models within one model because the sum of the gradients wrt losses gives the gradients wrt loss
            loss.backward()
            optimizer.step()
            epoch_losses += losses.detach()
            epoch_loss += loss.detach()
        if no_printing is False:
            print(f"Epoch {epoch}, train loss is {epoch_loss}")
    return epoch_losses


def load_models_section2(models: dict, big: bool = False):
    """Load the models for Section 2 saved during training
    models: empty dictionary in which to store the models
    big: boolean that indicates whether to load the small models or the big models"""
    assert models == {}
    sparsities = SPARSITIES

    config = Config_PaR(big=True) if big else Config_PaR()
    pathname = BIG_MODELS_PATHNAME if big else SMALL_MODELS_PATHNAME

    num_features = config.input_dim
    reduce_to_dim = config.hidden_dim
    importance = config.importance

    for sparsity in sparsities:
        model_filename = pathname + str(sparsity)
        if os.path.exists(model_filename):
            models[sparsity] = (
                ProjectAndRecover(num_features, reduce_to_dim, importance)
                .to(device)
                .train()
            )
            models[sparsity].load_state_dict(t.load(model_filename))
            models[sparsity].eval()
        else:
            raise ImportError



#  Train different sparsities and store models for Section 2
if __name__ == "__main__":
    pathname = SMALL_MODELS_PATHNAME  # SMALL_MODELS_PATHNAME / BIG_MODELS_PATHNAME
    config = Config_PaR(big=False)  # big = False / big = True

    num_features = config.input_dim
    reduce_to_dim = config.hidden_dim
    importance = config.importance

    sparsities = SPARSITIES

    size_trainingdata = 200000
    batch_size = 128
    epochs = 25

    datasets = {}
    trainloaders = {}
    models = {}

    for sparsity in sparsities:
        datasets[sparsity] = generate_synthetic_data(
            num_features, size_trainingdata, sparsity
        )
        trainloaders[sparsity] = DataLoader(
            tuple((datasets[sparsity])), batch_size=batch_size
        )
        model_filename = pathname + str(sparsity)
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