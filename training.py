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
SPARSITIES = [0., .7, .9, .97, .99, .997, .999]
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
    """Model architechture according to Section 2 of the paper
    
    weights: weight matrix of shape (hidden_features, input_features)
    bias: vector of length input_features
    importance: vector of length input_features, used as weights in the loss function
    """
    def __init__(self, input_features: int, hidden_features: int, importance: t.Tensor):
        super().__init__()
        #self.linear = t.nn.Linear(input_features, hidden_features, bias=False)
        self.weights = t.nn.Parameter(t.rand((hidden_features, input_features), requires_grad=True)*.8)
        self.bias = t.nn.Parameter(t.zeros((input_features), requires_grad=True))
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


class Config_PaR:
    """Configuration for the ProjectAndRecover model with two predefined configurations
    The default configuration is that of small models of Section 2 of the paper"""
    input_dim = 20
    hidden_dim = 5
    importance = t.tensor([.7 ** i for i  in range(input_dim)])
    def __init__(self, big: bool = False):
        """If big = True, initialize configuration of big models of Section 2"""
        if big:
            self.input_dim = 80
            self.hidden_dim = 20
            self.importance = t.tensor([.9 ** i for i  in range(self.input_dim)])


#%% Loss function
def weighted_MSE(x, x_hat, weights= 1) -> float:
    """Compute weighted MSE of x and x^hat wrt weights"""
    assert x.shape == x_hat.shape
    assert x.shape[-1] == weights.shape[-1]
    squared_error = (x - x_hat) ** 2
    weighted_squared_error = weights * squared_error
    return weighted_squared_error.mean()


#%% Training
def train(model: ProjectAndRecover, trainloader: DataLoader, epochs: int = 15, lr: float = 0.001) -> ProjectAndRecover:
    """Train model of class ProjectAndRecover on data provided in trainloader
    
    epochs: number of epochs to train
    lr: learning rate
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

#%% train one model
if __name__ == "__main__":
    
    input_dim = 20                   # 20 for small models / 80 for big models
    hidden_dim = 5                   # 5 for small models / 20 for big models
    importance_factor = .7           # .7 for small models / .9 for big models
    importance = t.tensor([importance_factor ** i for i  in range(input_dim)])
    
    sparsity = 0.07                   # or any float in [0,1)
    size_trainingdata = 100000
    data = generate_synthetic_data(input_dim, 100000, sparsity)

    batch_size = 128
    epochs = 25
    trainloader = DataLoader(tuple((data)), batch_size= batch_size)
    device = 'cpu'

    model = ProjectAndRecover(input_dim, hidden_dim, importance).to(device).train()
    model = train(model, trainloader, epochs=epochs)


#%% Train different sparsities and store models for Section 2
if __name__ == "__main__":
    pathname = SMALL_MODELS_PATHNAME       # SMALL_MODELS_PATHNAME / BIG_MODELS_PATHNAME 
    config = Config_PaR(big= False)        # big = False / big = True
    
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
        datasets[sparsity] = generate_synthetic_data(num_features, size_trainingdata, sparsity)
        trainloaders[sparsity] = DataLoader(tuple((datasets[sparsity])), batch_size= batch_size)
        model_filename = pathname + str(sparsity)
        if os.path.exists(model_filename):
            print("Loading model from disk: ", model_filename)
            models[sparsity] = ProjectAndRecover(num_features, reduce_to_dim, importance).to(device).train()
            models[sparsity].load_state_dict(t.load(model_filename))
        else:
            print("Training model from scratch")
            models[sparsity] = ProjectAndRecover(num_features, reduce_to_dim, importance).to(device).train()
            models[sparsity] = train(models[sparsity], trainloaders[sparsity], epochs = epochs)
            t.save(models[sparsity].state_dict(), model_filename)
        #plot_weights_and_bias(models[sparsity].weights.data, models[sparsity].bias.data)


#%% If necessary, train more
if __name__ == "__main__":
    i=4
    sparsity = sparsities[i]
    models[sparsity] = train(models[sparsity], trainloaders[sparsity], epochs=10, lr=0.0001)
    t.save(models[sparsity].state_dict(), pathname + str(sparsity))
    
#%%
def load_models_section2(models: dict, big: bool = False):
    """Load the models for Section 2 saved during training
    models: empty dictionary in which to store the models
    big: boolean that indicates whether to load the small models or the big models"""
    assert models == {}
    sparsities = SPARSITIES
    
    config = Config_PaR(big= True) if big else Config_PaR()   
    pathname = BIG_MODELS_PATHNAME if big else SMALL_MODELS_PATHNAME
    
    num_features = config.input_dim          
    reduce_to_dim = config.hidden_dim
    importance = config.importance    

    for sparsity in sparsities:
        model_filename = pathname + str(sparsity)
        if os.path.exists(model_filename):
            models[sparsity] = ProjectAndRecover(num_features, reduce_to_dim, importance).to(device).train()
            models[sparsity].load_state_dict(t.load(model_filename))
            models[sparsity].eval()
        else:
            raise ImportError
