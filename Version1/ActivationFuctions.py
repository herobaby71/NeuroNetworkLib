import numpy as np

class ActivationFunctions:
    def __init__(self):

    def sigmoid(self, z, gradient = False):
        if(gradient):
            return sigmoid(z)*(1-sigmoid(z))
        return 1/(1+np.exp(-z))

    #for tanh function:
    #if t =1: cost = -.5*log((y+1)/2)
    #if t =0: cost = -
    
    def tanh(self, z, gradient = False):
        if(gradient):
            return
        return 2/(1 + np.exp(-2 * x))) - ) 
