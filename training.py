#%%
import torch as t
import os

from typing import Union, Optional
from einops import rearrange, reduce, repeat
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from matplotlib.pyplot import matshow

SMALL_MODELS_PATHNAME = "./model-weights/section2-small/"
BIG_MODELS_PATHNAME = "./model-weights/section2-big/"
device = 'cpu'

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

#%% Model
class ProjectAndRecover(t.nn.Module):
    def __init__(self, input_features: int, hidden_features: int, importance: t.Tensor):
        super().__init__()
        #self.linear = t.nn.Linear(input_features, hidden_features, bias=False)
        self.weights = t.nn.Parameter(t.rand((hidden_features, input_features))*.5)
        self.bias = t.nn.Parameter(t.rand(input_features), requires_grad=True)
        self.relu = t.nn.ReLU()
        assert importance.shape == t.Size([input_features])
        self.importance = importance

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


#%% Training
def train(model: ProjectAndRecover, trainloader: DataLoader,input_dim:int, hidden_dim: int, epochs: int = 15, lr: float = 0.001) -> ProjectAndRecover:
    """trains model on data provided in trainloader
    """
    optimizer = t.optim.Adam(model.parameters(),lr=lr)
    for epoch in tqdm(range(epochs)):
        for i, x in enumerate(trainloader):
            x = x.to(device)
            # y = y.to(device)
            optimizer.zero_grad()
            x_hat = model(x)
            loss = weighted_MSE(x, x_hat, model.importance)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch}, train loss is {loss}")
    return model

#%% train one small model
if __name__ == "__main__":
    input_dim = 20
    hidden_dim = 5
    sparsity = 0.7
    importance = t.tensor([.7 ** i for i  in range(input_dim)])
    data = generate_synthetic_data(input_dim, 100000, sparsity)

    batch_size = 128
    trainloader = DataLoader(tuple((data)), batch_size= batch_size)
    device = 'cpu'

    model = ProjectAndRecover(input_dim, hidden_dim, importance).to(device).train()
    model = train(model, trainloader, input_dim, hidden_dim, epochs=20)

#%% train one big model
if __name__ == "__main__":
    input_dim = 80
    hidden_dim = 20
    sparsity = 0.999
    importance = t.tensor([.9 ** i for i  in range(input_dim)])
    data = generate_synthetic_data(input_dim, 100000, sparsity)

    batch_size = 128
    trainloader = DataLoader(tuple((data)), batch_size= batch_size)
    device = 'cpu'

    model = ProjectAndRecover(input_dim, hidden_dim, importance).to(device).train()
    model = train(model, trainloader, input_dim, hidden_dim, epochs=25)

#%% Train different sparsities
if __name__ == "__main__":
    MODELS_PATHNAME = BIG_MODELS_PATHNAME # SMALL_MODELS_PATHNAME / BIG_MODELS_PATHNAME 

    num_features = 80             # 20 for small models / 80 for big models
    reduce_to_dim = 20            # 5 for small models / 20 
    size_trainingdata = 100000
    batch_size = 128
    epochs = 25
    importance_factor = .9        # .7 for small models / .9 for big models

    sparsities = [0., .7, .9, .97, .99, .997, .999]
    importance = t.tensor([importance_factor ** i for i  in range(num_features)])        

    datasets = {}
    trainloaders = {}
    models = {}
    for sparsity in sparsities:
        datasets[sparsity] = generate_synthetic_data(num_features, size_trainingdata, sparsity)
        trainloaders[sparsity] = DataLoader(tuple((datasets[sparsity])), batch_size= batch_size)
        model_filename = MODELS_PATHNAME + str(sparsity)
        if os.path.exists(model_filename):
            print("Loading model from disk: ", model_filename)
            models[sparsity] = ProjectAndRecover(num_features, reduce_to_dim, importance).to(device).train()
            models[sparsity].load_state_dict(t.load(model_filename))
        else:
            print("Training model from scratch")
            models[sparsity] = ProjectAndRecover(num_features, reduce_to_dim, importance).to(device).train()
            models[sparsity] = train(models[sparsity], trainloaders[sparsity], input_dim=num_features, hidden_dim=reduce_to_dim, epochs = epochs)
            t.save(models[sparsity].state_dict(), model_filename)
        #plot_weights_and_bias(models[sparsity].weights.data, models[sparsity].bias.data)


#%% If necessary, train more
if __name__ == "__main__":
    i=4
    sparsity = sparsities[i]
    models[sparsity] = train(models[sparsity], trainloaders[sparsity], input_dim=num_features, hidden_dim=reduce_to_dim, epochs=10, lr=0.0001)
    t.save(models[sparsity].state_dict(), MODELS_PATHNAME + str(sparsity))
    

