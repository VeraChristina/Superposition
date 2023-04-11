#%%
import torch as t
from typing import Union, Optional
from einops import rearrange, reduce, repeat
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


import matplotlib as mlp
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from matplotlib.pyplot import matshow


#%% Data generation
def generate_synthetic_data(num_features: int, size: int = 100000, sparsity: float = 1.0) -> t.Tensor:
    """ generates synthetic data with given feature sparsity and importance
        input: 
        num_features    number of features (i.e. the dimension of the synthetic data)
        size            number of data points to be generated
        sparsity        float in [0,1] that indicates how sparsely features are distributed
        
    
        output: tensor A of shape (size, num_features), 
                each column is a synthetic data point, i.e. a vector of length num_features
                each feature is zero with probability (1-sparsity) and otherwise uniformly distributed in [0, 1]
    """
    assert 0 <= sparsity <= 1
    mask = (t.rand(tuple((size, num_features))) < 1 - sparsity )
    A = t.rand(size=(size, num_features)) * mask
    return A

sparsity = 0.7
data = generate_synthetic_data(30, 500000, sparsity) 
# %% Test Data


def test_data_sparsity(data, target_sparsity):
    """test wether each feature in data is sparsely distributed, i.e. zero with probability sparsity"""
    data = (data == t.tensor([0]))
    empiric_sparsity = reduce(data, 'b f -> f', 'sum') / data.shape[0]
    allclose(empiric_sparsity, t.ones(data.shape[1]) * target_sparsity, .01)

def assert_shape_equal(actual: t.Tensor, expected: t.Tensor) -> None:
    if actual.shape != expected.shape:
        raise AssertionError(f"expected shape={expected.shape}, got {actual.shape}")


def allclose(actual: t.Tensor, expected: t.Tensor, rtol=1e-4) -> None:
    assert_shape_equal(actual, expected)
    left = (actual - expected).abs()
    right = rtol * expected.abs()
    num_wrong = (left > right).sum().item()
    if num_wrong > 0:
        print(f"Test failed. Max absolute deviation: {left.max()}")
        print(f"Actual:\n{actual}\nExpected:\n{expected}")
        raise AssertionError(f"allclose failed with {num_wrong} / {left.nelement()} entries outside tolerance")
    else:
        print(f"Test passed with max absolute deviation of {left.max()}")

test_data_sparsity(data, sparsity)


#%% Model
class ProjectAndRecover(t.nn.Module):
    def __init__(self, input_features: int, hidden_features : int):
        super().__init__()
        #self.linear = t.nn.Linear(input_features, hidden_features, bias=False)
        self.weights = t.nn.Parameter(t.rand((hidden_features, input_features)))
        self.bias = t.nn.Parameter(t.rand(input_features), requires_grad=True)
        self.relu = t.nn.ReLU()
        
    def forward(self, x: t.Tensor) -> t.Tensor:
        """ Reduce via linear transformation, recover via its transpose, add bias and apply ReLU
        
        x: shape (batch, input_features)
        Return: shape (batch, input_features)
        """
        x = t.einsum('h i, b i -> b h', self.weights, x)
        x = t.einsum('h i, b h -> b i', self.weights, x) + self.bias
        return self.relu(x)


#%% Loss function
def weighted_MSE(x, x_hat, weights= 1) -> float:
    assert x.shape == x_hat.shape
    assert x.shape[-1] == weights.shape[-1]
    squared_error = (x - x_hat) ** 2
    
    weighted_squared_error = weights * squared_error
    return weighted_squared_error.mean()


# x = t.tensor([1, 2, 3])
# x_hat = t.tensor([1.1, 2.2, 3.3])
# weights = t.tensor([.1, .2, .3])
# weighted_MSE(x, x_hat, weights=weights)


#%% Training
def train(trainloader: DataLoader,input_dim:int, hidden_dim: int, epochs: int, model: Optional[ProjectAndRecover]=None) -> ProjectAndRecover:
    """trains model on data provided in trainloader; if no model is provided a PojectAndRecover model is initialized and trained
    """
    if model is None:
        model = ProjectAndRecover(input_dim, hidden_dim).to(device).train()
    optimizer = t.optim.Adam(model.parameters(),lr=.001)
    for epoch in tqdm(range(epochs)):
        for i, x in enumerate(trainloader):
            x = x.to(device)
            # y = y.to(device)
            optimizer.zero_grad()
            x_hat = model(x)
            loss = weighted_MSE(x, x_hat, importance)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch}, train loss is {loss}")
        #print(f"Saving model to: {os.path.abspath(MODEL_FILENAME)}")
        #t.save(model, MODEL_FILENAME)
    return model

# %% 
input_dim = 20
hidden_dim = 5
sparsity = 0.
importance = t.tensor([.7 ** i for i  in range(input_dim)])
data = generate_synthetic_data(20, 100000, sparsity)

batch_size = 128
trainloader = DataLoader(tuple((data)), batch_size= batch_size)
device = 'cpu'

model = train(trainloader, input_dim = input_dim, hidden_dim=hidden_dim, epochs=25)

#%% Heat maps
W = model.weights.data
b = model.bias.data
def plot_weights_and_bias(W, b):
    fig = plt.figure(figsize=(5.5, 5))
    grid = ImageGrid(fig, 111,  
                     nrows_ncols=(1, 2),
                     axes_pad=0.2
                     )
    for ax, im in zip(grid, [W.T @ W , b.reshape((len(b), 1))]):
        ax.imshow(im, origin="upper", vmin= -1, vmax= 1, cmap=mlp.colormaps['PiYG'])
    grid[0].set_title(f'Weight matrix and bias for sparsity {sparsity}')
    plt.show()
plot_weights_and_bias(W, b)
#%% if necessary, continue training
model = train(trainloader, input_dim, hidden_dim, epochs=10, model=model)

#%% Visualization
def compute_superposition(W) -> list[t.Tensor]:
    num_features = W.shape[1]
    matrix = W.T @ W
    representation = t.einsum('ij, kj -> j', matrix, matrix) ** .5
    superposition = t.einsum('ij, kl -> j', matrix, matrix)
    superposition -= representation ** 2
    return (superposition, representation)

   
        
def visualize_superposition(W):
    representation, superposition = compute_superposition(model.weights)
    
    fig, ax = plt.subplots()
    features = range(num_features) 
    bars = representation.detach().numpy()
    bars = bars / bars.max()
    color_values = superposition.detach().numpy()
    color_values = color_values / color_values.max()
    cmap = mlp.colormaps['cividis_r']
    bar_colors = [cmap(color) for color in color_values]
    
    ax.invert_yaxis()
    ax.barh(features, bars, color=bar_colors)
    ax.set_ylabel('features')
    ax.set_box_aspect(2)

    plt.show()
        
visualize_superposition(model.weights)

#%% Train different sparsities
num_features = 20
reduce_to_dim = 5
size_trainingdata = 100000
batch_size = 128

sparsities = [0., .7, .9, .97, .99, .997, .999]
datasets = {}
trainloaders = {}
models = {}
for sparsity in sparsities:
    datasets[sparsity] = generate_synthetic_data(num_features, size_trainingdata, sparsity)
    trainloaders[sparsity] = DataLoader(tuple((datasets[sparsity])), batch_size= batch_size)
    models[sparsity] = train(trainloaders[sparsity], input_dim=num_features, hidden_dim=reduce_to_dim, epochs = 15)
    plot_weights_and_bias(models[sparsity].weights.data, models[sparsity].bias.data)
#%%
models[0.] = train(trainloaders[0.], input_dim=num_features, hidden_dim=reduce_to_dim, epochs=10, model=models[0.])
plot_weights_and_bias(models[0.].weights.data, models[0.].bias.data)
# %%
fig = plt.figure(figsize=(14.4, 2))
grid = ImageGrid(fig, 111,  
                nrows_ncols=(1, 14),
                axes_pad=0.2
                )
plotpairs =[]
for sparsity in sparsities:
    W,b = models[sparsity].weights.data, models[sparsity].bias.data
    plotpairs += [W.T @ W, b.reshape((len(b), 1))]
for ax, im in zip(grid, plotpairs):
    ax.imshow(im, origin="upper", vmin= -1, vmax= 1, cmap=mlp.colormaps['PiYG'])
    ax.set_label(f'Weight matrix and bias for sparsity {sparsity}')
plt.show()
# %%
