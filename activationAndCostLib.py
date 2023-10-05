# modules

import numpy as np

# gradient functions of costs

def crossEntropy(x, y):
    """
    gradient of cost for cross-entropy cost function C(x,y) = -y*log(x)-(1-y)*log(1-x)
    :param x: output calculated by NN
    :param y: true output
    """
    return (1 - y) / (1 - x) - (y / x)

def MSE(x, y):
    """
    gradient of cost for cost function C = 0.5*(x-y)**2
    :param x: output calculated by NN
    :param y: true output
    """
    return x - y

# activation functions

def sigmoid(z):
    # sigmoid function
    return 1.0 / (1.0 + np.exp(-z))

def diffSigmoid(z):
    # derivative of sigmoid function
    return np.exp(-z) / ((1.0 + np.exp(-z)) ** 2)

def relu(z):
    # rectified linear unit function
    return np.maximum(z,0)

def diffRelu(z):
    # derivative rectified linear unit function
    return z>=0
