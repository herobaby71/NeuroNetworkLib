import numpy as np
from scipy.special import expit
from ActivationFunctions import ActivationFunctions

def InitializeWeights(input_layer_size, out_layer_size):        
    return np.sqrt(2/(input_layer_size+out_layer_size))*np.random.randn(out_layer_size,input_layer_size+1)

class Layers:
    """layers w/ activation functions """
    def __init__(self, X = None, layer_name = "affine"):
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
    
    def forwardProp(self,X = None, weights = None, bias = None, conv_params = None, maxpool_params = None):
        """this is the forwardprop general call to other methods in this class"""
        if(X is None): X = self.cache[0]
        if(weights is None): return #can't forward prop without weights
        func_name = self.name + "_forward"
        if(self.getName() == 'conv'):
            getattr(self, func_name)(
                X, weights, bias, conv_params["pad"], conv_params["stride"])
        if(self.getName() == 'max_pooling'):
            getattr(self, func_name)(
                X, weights, maxpool_params["H"], maxpool_params["W"], maxpool_params["stride"])
        return getattr(self, func_name)(X, weights)
    
    def backwardProp(self, deltaOut = None, cache= None):
        """this is the backwardprop general call to other methods in this class"""
        if(cache is None): cache = self.cache
        func_name = self.name + "_backward"
        return getattr(self, func_name)(deltaOut, cache)

    def affine_forward(self, X, weights, *args):
        """ fully connected forward pass (linear) """
        m = X.shape[0]
        a0 = np.append(np.ones([m,1]),X, axis = 1)
        z1 = np.dot(a0, weights.T)
        self.cache = (X, weights)
        return z1, self.cache
    
    def affine_backward(self, deltaOut, cache = None, *args):
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

    def check_conv_params(self, filH, filW):
        if(not(filH==filW)): return False
        if(filH%2 == 0 or filW%2==0): return False
        return True
    
    def conv_forward(self, X, weights, bias, pad = 0, stride = 1):
        """
         X: (m, channels, heights, widths)
         weights (bunch of filters params): (n,channels, height, width)
        """
        m, c, h, w = X.shape
        n_fil, _, h_fil, w_fil = weights.shape
        
        #output dims
        out_h = (h-h_fil+2*pad)/stride + 1 
        out_w = (w-w_fil+2*pad)/stride + 1

        out = np.zeros((m, n_fil, out_h, out_w))
        for l in range(m):
            #padding
            pad_dims = [(0,0),[pad,pad],[pad,pad]]
            img = np.pad(X[l,:], pad_dims, 'constant', constant_values=(0,0))
            for k in range(n_fil): #depths
                for j in range(out_h): #height
                    for i in range(out_w): #width
                        img_parse = img[:,stride*j:h_fil+ stride*j, stride*i: w_fil+ stride*i]
                        out[l,k,j,i] = np.sum(img_parse*weights[k, :, :, :]) + bias[k]

        self.cache = (X, weights, bias, pad, stride)
        return (out, self.cache)

    
    def conv_backward(self, deltaOut, cache):
        """
           Convolutional layer backward propagation
        """
        X,weights, bias, pad, stride = cache

        m, c, h, w = X.shape
        _, _, h_fil, w_fil = weights.shape

        _, n_fil, h_out, w_out = deltaOut.shape

        deltaX = np.zeros(X.shape)
        deltaWeights = np.zeros(weights.shape)
        deltaBias = np.zeros(bias.shape)
        for l in range(m):
            pad_dims = [(0,0),[pad,pad],[pad,pad]]
            img = np.pad(X[l,:], pad_dims, 'constant', constant_values=(0,0))
            deltaX_img = np.zeros(img.shape)
            for k in range(n_fil): #loop through the depths
                for j in range(h_out): #height
                    for i in range(w_out): #width
                        deltaX_pad[:,stride*j:h_fil+ stride*j, stride*i: w_fil+ stride*i] = weights[k,:,:,:]*deltaOut[l,k,j,i]
                        deltaWeights[k,:,:,:] += img[:,stride*j:h_fil+ stride*j, stride*i: w_fil+ stride*i]*deltaOut[l,k,j,i]
                        deltaBias+= deltaOut[l,k,j,i]
            deltaX[l,:,:,:] = deltaX_pad 
        return (deltaX, deltaWeights, deltaBias)

    def max_pooling_forward(self, X, poolH, poolW, stride, *args):
        """
           max pooling forward pass 
        """
        m, c, h, w = X.shape

        #output dims
        out_h = (h-poolH)/stride + 1
        out_w = (w-poolW)/stride + 1
        out = np.zeros((m,c,out_h,out_w))
        for k in range(m):
            for j in range(out_h):
                for i in range(out_w):
                    img_parse = X[k,:,stride*j:poolH+stride*j, stride*i: poolW+stride*i]
                    img_parse = np.reshape(img_parse, (c, poolH*poolW))#combine w,h into rows
                    out[k, :, j, i] = np.max(img_parse, axis = 1)
        self.cache = (X, poolH, poolW, stride)
        return out, self.cache
    
    def max_pooling_backward(self, deltaOut, cache, *args):
        X, poolH, poolW, stride = cache

        m, c, h, w = X.shape
        _, _, h_out, w_out = deltaOut.shape

        deltaX = np.zeros(X.shape)
        for k in range(m):
            for j in range(h_out):
                for i in range(w_out):
                    img_parse = X[k,:,stride*j:poolH+stride*j, stride*i: poolW+stride*i]
                    img_parse = np.reshape(img_parse, (c, poolH*poolW))
                    zeros_img_parse = np.zeros(img_parse.shape)
                    zeros_img_parse[np.argmax(img_parse, axis = 1)] = 1
                    deltaX[n,:, stride*j:poolH+stride*j, stride*i: poolW+stride*i] = np.reshape(zeros_img_parse, (c, poolH,poolW))
        return deltaX

    
    def Elu_forward(self):
        pass
    
    def Elu_backward(self):
        pass
    
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
    
    def affine_LReLU_forward(self, X = None, weights = None, *args):
        z1, cache_af= self.affine_forward(X, weights)
        a1 = self.ReLU_forward(z1)
        cache_LReLU = z1 # save the input matrix to LRelu for backprop computation
        cache = (cache_af, cache_LReLU)
        return (a1, cache)
    
    def affine_LReLU_backward(self,deltaOut, cache):
        cache_af, cache_rl = cache
        x, weights = cache_af
        deltaRe = self.LReLU_backward(deltaOut, cache_rl)
        deltaX, deltaWeights  = self.affine_backward(deltaRe, cache_af)
        return (deltaX, deltaWeights)
    
    def sigmoid_forward(self, X, *args):
        self.activFunc = ActivationFunctions("sigmoid")
        self.cache = (X, None)
        return self.activFunc.getVal(X, False)
        
    def sigmoid_backward(self, X, cache, *args):
        return self.activFunc.getVal(X, True)

    def sigmoid_loss(self, x = None, y = None, optimizing = True, *args):
        """ Compute sigmoid loss function and also the delta difference between the target and y"""
        self.activFunc = ActivationFunctions("sigmoid")
        m = x.shape[0]
        probs = self.activFunc.getVal(x)

        if(not optimizing): return probs    
        
        J = (-1/m)*np.sum((y*np.log(a2)) + (1-y)*(np.log(1-a2)))

        delta = probs - y
        return (J, delta)

    def softmax_loss(self,x = None, y= None, optimizing = True, *args):
        """ Compute Softmax loss function and also the delta difference between the target and y"""
        self.activFunc = ActivationFunctions("softmax")
        m = y.shape[0]
        probs = self.activFunc.getVal(x)

        if(not optimizing): return probs

        J = np.log(y*probs)
        J[np.isneginf(J)] = 0
        J = (-1/m)*np.sum(J)
        
        delta = probs - y
        return (J, delta)

    def svm_loss(self):
        return


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
