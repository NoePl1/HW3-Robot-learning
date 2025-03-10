from typing import Union

import torch
from torch import nn

Activation = Union[str, nn.Module]

_str_to_activation = {
    'relu': nn.ReLU(),
    'tanh': nn.Tanh(),
    'leaky_relu': nn.LeakyReLU(),
    'sigmoid': nn.Sigmoid(),
    'selu': nn.SELU(),
    'softplus': nn.Softplus(),
    'identity': nn.Identity(),
}
    
def build_mlp(
        input_size: int,
        output_size: int,
        **kwargs
    ):
    """
    Builds a feedforward neural network

    arguments:
    n_layers: number of hidden layers
    size: dimension of each hidden layer
    activation: activation of each hidden layer
    input_size: size of the input layer
    output_size: size of the output layer
    output_activation: activation of the output layer

    returns:
        MLP (nn.Module)
    """
    try:
        params = kwargs["params"]
    except:
        params = kwargs

    if isinstance(params["output_activation"], str):
        output_activation = _str_to_activation[params["output_activation"]]

    # TODO: return a MLP. This should be an instance of nn.Module
    # Note: nn.Sequential is an instance of nn.Module.
    n_layers = len(params["layer_sizes"])
    sizes = params["layer_sizes"]
    activations = []
    for activation in params["activations"]:
        if isinstance(activation, str):
            activations.append(_str_to_activation[activation])
    mlp = nn.Sequential()
    mlp.add_module("fc1", nn.Linear(input_size, sizes[0]))
    mlp.add_module("activation1", activations[0])
    for i in range(n_layers-1):
        mlp.add_module(f'fc{i+2}', nn.Linear(sizes[i], sizes[i+1]))
        mlp.add_module(f'activation{i+2}', activations[i+1])
    mlp.add_module("output",nn.Linear(sizes[-1], output_size))
    mlp.add_module("output_activation", output_activation)
    return mlp

device = None


def build_cnn(
        input_size: int,
        output_size: int,
        **kwargs
):
    """
    Builds a feedforward neural network

    arguments:
    n_layers: number of hidden layers
    size: dimension of each hidden layer
    activation: activation of each hidden layer
    input_size: size of the input layer
    output_size: size of the output layer
    output_activation: activation of the output layer

    returns:
        MLP (nn.Module)
    """
    try:
        params = kwargs["params"]
    except:
        params = kwargs

    if isinstance(params["output_activation"], str):
        output_activation = _str_to_activation[params["output_activation"]]

    # TODO: return a MLP. This should be an instance of nn.Module
    # Note: nn.Sequential is an instance of nn.Module.

    cnn = nn.Sequential()
    cnn.add_module("conv1", nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1))
    cnn.add_module("bn1", nn.BatchNorm2d(32))
    cnn.add_module("relu1", nn.ReLU())
    cnn.add_module("maxpool1", nn.MaxPool2d(kernel_size=2, stride=2))

    cnn.add_module("conv2", nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1))
    cnn.add_module("bn2", nn.BatchNorm2d(64))
    cnn.add_module("relu2", nn.ReLU())
    cnn.add_module("maxpool2", nn.MaxPool2d(kernel_size=2, stride=2))

    cnn.add_module("conv3", nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1))
    cnn.add_module("bn3", nn.BatchNorm2d(128))
    cnn.add_module("relu3", nn.ReLU())
    cnn.add_module("maxpool3", nn.MaxPool2d(kernel_size=2, stride=2))

    cnn.add_module("flatten", nn.Flatten())
    cnn.add_module("fc", nn.Linear(128 * 8 * 8, output_size))

    return cnn


device = None

def init_gpu(use_gpu=True, gpu_id=0):
    global device
    if torch.cuda.is_available() and use_gpu:
        device = torch.device("cuda:" + str(gpu_id))
        print("Using GPU id {}".format(gpu_id))
    else:
        device = torch.device("cpu")
        print("GPU not detected. Defaulting to CPU.")


def set_device(gpu_id):
    torch.cuda.set_device(gpu_id)

def from_numpy(*args, **kwargs):
    return torch.from_numpy(*args, **kwargs).float().to(device)

def to_numpy(tensor):
    return tensor.to('cpu').detach().numpy()
