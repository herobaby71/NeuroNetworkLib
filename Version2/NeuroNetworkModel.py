import numpy as np
from Layers import Layers
import pandas as pd
from Optimization import Optimizer
from sklearn.model_selection import train_test_split

class FeedForwardModel:
    def __init__(self, X = np.array([]), y = np.array([]), lamb = 0):
        self.X = X
        self.y = y
        self.X_train,self.X_cv,self.y_train,self.y_cv = train_test_split(X,y,test_size = .15,  random_state=42)
        self.lamb = lamb
        self.layers = {}
        self.layers_count = 0

        #Optimizer
        self.optimizer = Optimizer()
        
    def addLayer(self, name = "affine"):
        self.layers[self.layers_count] = Layers(X = self.X, layer_name = name)
        self.layers_count +=1

    def addConvLay(self):
        pass
    def costFunction(self,  model = None, X = None, y = None, optimizing = True):
        """
           Return the cost (loss) and the gradient for optimization purposes
           model: Weights params "W0", "W1" ... "WL"
           optimizing: if True, then compute the grads. if False, only compute forward propagation
        """        
        if(model is None): return # do nothing since weights is epty
        if(X is None):
            X = self.X
        if(y is None):
            y = self.y

        #m: number of examples
        #lamb: regularization prams
        m = X.shape[0]
        lamb = self.lamb
        #forward propagation
        outputs = [None] * (self.layers_count+1)
        outputs[-1] = X
        caches = [None] * (self.layers_count)
        deltas = [None] * (self.layers_count)
        grads = {}
        for i in range(self.layers_count-1):
            outputs[i], caches[i] = self.layers[i].forwardProp(outputs[i-1], model["W" + str(i)])

        #if we want to predict and not train a model, we want only the probability of the outputs
        if(not(optimizing)):
            return self.layers[self.layers_count-1].getCost(outputs[self.layers_count-2], y, optimizing)

        #compute Cost function
        J, deltas[self.layers_count-1] = self.layers[self.layers_count-1].getCost(outputs[self.layers_count-2], y)

        #regularization
        reg = 0
        for i in range(len(model)):
            reg+=np.sum(np.power(model["W" + str(i)][:,1:],2))
        J+= (self.lamb/(2*m))*reg

        #backward propagation
        for i in reversed(range(self.layers_count-1)):
            deltas[i], grads["W"+str(i)] = self.layers[i].backwardProp(deltas[i+1], caches[i])        
            grads["W"+str(i)] *= 1/m
            grads["W"+str(i)][:,1:] = grads["W"+str(i)][:,1:] + (lamb/m)*model["W"+str(i)][:,1:]
        return J, grads
    
    def train(self, X, y, X_val = None, y_val = None, model = None, update_method = 'sgd', epochs = 30):  
        return self.optimizer.train(self.X, self.y, self.X_cv, self.y_cv, model, self.costFunction, update = update_method, numEpochs = epochs)


##def model():
##    weights = {}
##    weights["W0"] = InitializeWeights(784, 500)
##    weights["W1"] = InitializeWeights(500,10)
##    return weights
##
##X,y = openData()
##weights = openThetas()
##digit_recognizer = FeedForwardModel(X,y,0)
##digit_recognizer.addLayer("affine_ReLU")
##digit_recognizer.addLayer("affine")

##J, grads = digit_recognizer.costFunction(model = weights)
