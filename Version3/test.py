import numpy as np
from Layers import FeedForwardLayers as ffl
from Layers import InitializeWeights

activationFuncs = {0: "sigmoid", 1: "ReLU", 2: "LReLU", 3: "elu", 4: "softmax"}
def model():
    model = {}
    model["W1"] = InitializeWeights(784,500)
    model["W2"] = InitializeWeights(500, 10)
    return model

def FeedForwardNN(X, y = None, numLayers  = 3, model, lamb = 0):
    W1, W2 = model['W1'], model['W2']
    m, n = X.shape
    #if y is None, then predict given X and return the value
        
    layer1= ffl(n, X, y, W1, "sigmoid")
    a1 = layer1.forwardProp()
    layer2 = ffl(n, a1, y, W2, "softmax")
    J, a2 = layer2.forwardProp()

    d2 = layer2.backwardProp(a2)
    d1 = layer1.backwardProp(d2)
