import numpy as np
from scipy.special import expit
from ActivationFunctions import ActivationFunctions

def InitializeWeights(input_layer_size, out_layer_size):        
    return np.sqrt(2/(input_layer_size+out_layer_size))*np.random.randn(out_layer_size,input_layer_size+1)

class Layers:
    """layers w/ activation functions """
    def __init__(self, X = None, layer_name = "affine" ):
        self.name = layer_name
        self.cache = (X, None) #save neccessary data to compute backward prop
        self.activFunc = None #save the activation function used
    def getName(self):
        return self.name
    def getCost(self, x, y = None, optimizing = True):
        if(y is None): return
        if(self.name == "softmax" or self.name == "svm" or self.name == "sigmoid"):
            func_name = self.name + "_loss"
            return getattr(self, func_name)(x, y, optimizing)
        return
        
    def forwardProp(self,X = None, weights = None):
        """this is the forwardprop general call to other methods in this class"""
        if(X is None): X = self.cache[0]
        if(weights is None): return #can't forward prop without weights
        func_name = self.name + "_forward"
        return getattr(self, func_name)(X, weights)
    
    def backwardProp(self, deltaOut = None, cache= None):
        """this is the backwardprop general call to other methods in this class"""
        if(cache is None): cache = self.cache
        func_name = self.name + "_backward"
        return getattr(self, func_name)(deltaOut, cache)

    def affine_forward(self, X, weights):
        """ fully connected forward pass (linear) """
        m = X.shape[0]
        a0 = np.append(np.ones([m,1]),X, axis = 1)
        z1 = np.dot(a0, weights.T)
        self.cache = (X, weights)
        return z1, self.cache
    
    def affine_backward(self, deltaOut, cache = None):
        """ fully connected backward pass """
        if(cache is None):
            cache = self.cache    
        X, weights = cache
        m = X.shape[0]
        deltaX, deltaWeights = None, None

    
        deltaX = np.dot(deltaOut, weights[:,1:])
        #add bias to X before computing the gradient of weights
        deltaWeights = np.dot(deltaOut.T,np.append(np.ones([m,1]),X, axis = 1))
        return deltaX, deltaWeights
    
    def ReLU_forward(self, X, *args):
        """ ReLU layer forward"""
        self.activFunc = ActivationFunctions("ReLU")
        self.cache = (X, None)
        return self.activFunc.getVal(X)
    
    def ReLU_backward(self, X, cache, *args):
        """ ReLU layer backward """
        return ActivationFunctions("ReLU").getVal(X, True, cache)

    def LReLU_forward(self, X, *args):
        """ Leaky ReLU layer forward"""
        self.activFunc = ActivationFunctions("LReLU")
        self.cache = (X, None)
        return self.activFunc.getVal(X)
    
    def LReLU_backward(self, X, cache, *args):
        """ Leaky ReLU layer backward """
        return ActivationFunctions("LReLU").getVal(X, True, cache)
    
    def affine_ReLU_forward(self, X = None, weights = None, *args):
        """ combining both fully connected and ReLU """

        z1, cache_af= self.affine_forward(X, weights)
        a1 = self.ReLU_forward(z1)
        cache_ReLU = z1 # save the input matrix to Relu for backprop computation
        cache = (cache_af, cache_ReLU)
        return (a1, cache)
    
    def affine_ReLU_backward(self, deltaOut, cache):
        cache_af, cache_rl = cache
        x, weights = cache_af
        deltaRe = self.ReLU_backward(deltaOut, cache_rl)
        deltaX, deltaWeights  = self.affine_backward(deltaRe, cache_af)
        return (deltaX, deltaWeights)
    
    def affine_LReLU_forward(self):
        pass
    def affine_LReLU_backward(self):
        pass
    def sigmoid_forward(self, X, *args):
        self.activFunc = ActivationFunctions("sigmoid")
        self.cache = (X, None)
        return self.activFunc.getVal(X, False)
        
    def sigmoid_backward(self, X, cache, *args):
        return self.activFunc.getVal(X, True)

    def sigmoid_loss(self, x = None, y = None, optimizing = True, *args):
        self.activFunc = ActivationFunctions("sigmoid")
        m = x.shape[0]
        probs = self.activFunc.getVal(x)

        if(not optimizing): return probs    
        
        J = (-1/m)*np.sum((y*np.log(a2)) + (1-y)*(np.log(1-a2)))

        deltaX = probs - y
        return (J, deltaX)

    def softmax_loss(self,x = None, y= None, optimizing = True, *args):
        self.activFunc = ActivationFunctions("softmax")
        m = y.shape[0]
        probs = self.activFunc.getVal(x)

        if(not optimizing): return probs

        J = np.log(y*probs)
        J[np.isneginf(J)] = 0
        J = (-1/m)*np.sum(J)
        
        deltaX = probs - y
        return (J, deltaX)

class ConvLayer:
    pass

def main():
    weights1 = InitializeWeights(5,3)
    weights2 = InitializeWeights(3,3)
    X = np.array([[1,9,8,6,5],[1,3,8,6,4],[1,7,5,7,1],[1,6,9,7,9]])
    y = np.array([[0,1,0],[1,0,0],[0,0,1],[1,0,0]])
    l1 = Layers(layer_name = "affine")
    z1, cache = l1.forwardProp(X, weights1)
  
    l2 = Layers(layer_name = "ReLU")
    a1 = l2.forwardProp(z1)

    
    l3 = Layers(layer_name = "affine")
    z2, cache2 = l3.forwardProp(a1, weights2)

    l4 = Layers(layer_name = "softmax")
    J, delta = l4.getCost(z2, y)
    print(J)

##main()
