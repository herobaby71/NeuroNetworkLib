import numpy as np
from scipy.special import expit
class ActivationFunctions:
    def __init__(self,name):
        self.name = name
        self.cache = None
        
    def getName(self):
        return self.name
    
    def getVal(self, z, gradient = False, cache = None):
        if (cache is None):
            return getattr(self, self.name)(z, gradient)
        return getattr(self, self.name)(z, gradient, cache)
    #sigmoid
    def sigmoid(self, z, gradient = False):
        if(gradient):
            return self.sigmoid(z)*(1-self.sigmoid(z))
        #return 1/(1+np.exp(-z))
        self.cache = z
        return expit(z)

    #for derivative, if x > 0, return 1, O/W .01 () or 0
    def ReLU(self, z, gradient = False, cache = None):
        if(gradient):
            if(cache is None): cache = self.cache
            dz = np.array(z, copy = True)
            dz[cache <=0] = 0
            return dz
        self.cache = z
        return np.maximum(0,z)
    

    #Leaky ReLU with alpha (PReLus), to fix the dying ReLU problems
    def LReLU(self, z, gradient = False, alpha = .01, cache = None):
        if (gradient):
            if(cache is None):
                cache = self.cache
            dz = np.array(z, copy = True)
            dz[cache <=0] = alpha
            return dz
        self.cache = z
        temp = np.maximum(0,z).astype(bool) # False if <= 0
        z2 = np.array(z, copy = True)
        z2[temp==False] = alpha*z2[temp == False]
        return z2
    
    def elu(z, gradient = False, alpha = 1.0):
        """ ELUs lead to faster learning, with better generalization than ReLU,  while alleviate the vanishing gradient problem"""
        if(gradient):
            dz = np.array(z, copy = True)
            dz[self.cache < 0] = elu(self.cache)[self.cache < 0] + alpha
            return dz
        self.cache = z
        temp = np.maximum(0,z).astype(bool) # False if <=0
        z2 = np.array(z, copy = True)
        z2[temp == False] = alpha*(np.exp(z2[temp == False])-1)
        return z2
    #for the output layer to compute probability and the loss
    def softmax(self, z, gradient = False):
        if(gradient):
            return self.softmax(z)*(1-self.softmax(z))
        #subtract the maximum value to create negative value, which works better for exponents
        probs = np.exp(z - np.max(z, axis=1, keepdims=True))
        probs /= np.sum(probs, axis=1, keepdims=True)
        
        return probs

