import numpy as np
import time
from scipy.special import expit
from ActivationFunctions import ActivationFunctions
try:
  from cs231n.im2col_cython import col2im_cython, im2col_cython
  from cs231n.im2col_cython import col2im_6d_cython
except ImportError:
  print ('run the following from the cs231n directory and try again:')
  print ('python setup.py build_ext --inplace')
  print ('You may also need to restart your iPython kernel')

def InitializeWeights(input_layer_size, out_layer_size):        
    return np.sqrt(2/(input_layer_size+out_layer_size))*np.random.randn(out_layer_size,input_layer_size+1)

class Layers:
    """layers w/ activation functions """
    def __init__(self, X = None, layer_name = "affine", conv_params = None, maxpool_params = None):
        self.name = layer_name
        self.cache = (X, None) #save neccessary data to compute backward prop
        self.activFunc = None #save the activation function used
        self.conv_params = conv_params
        self.maxpool_params = maxpool_params
    def getName(self):
        return self.name
    def getCost(self, x, y = None, optimizing = True):
        if(y is None): return
        if(self.name == "softmax" or self.name == "svm" or self.name == "sigmoid"):
            func_name = self.name + "_loss"
            return getattr(self, func_name)(x, y, optimizing)
        return
    
    def forwardProp(self,X = None, weights = None, bias = None):
        """this is the forwardprop general call to other methods in this class"""
        if(X is None): X = self.cache[0]
        if(weights is None): return #can't forward prop without weights
        func_name = self.name + "_forward"
        if(self.getName() == 'conv' or self.getName() == 'conv_vec'):
            getattr(self, func_name)(
                X, weights, bias, self.conv_params["pad"], self.conv_params["stride"])
        if(self.getName() == 'max_pooling'):
            getattr(self, func_name)(
                X, weights, self.maxpool_params["H"], self.maxpool_params["W"], self.maxpool_params["stride"])
        return getattr(self, func_name)(X, weights)
    
    def backwardProp(self, deltaOut = None, cache= None):
        """this is the backwardprop general call to other methods in this class"""
        if(cache is None): cache = self.cache
        func_name = self.name + "_backward"
        return getattr(self, func_name)(deltaOut, cache)

    def affine_forward(self, X, weights, *args):
        """ fully connected forward pass (linear) """
        m = X.shape[0]
        n = np.prod(X.shape[1:])
        X2Dims = np.reshape(X, (m,n)) #expand to support more than 2 dimensions inputs
        
        a0 = np.append(np.ones([m,1]),X2Dims, axis = 1)
        z1 = np.dot(a0, weights.T)
        self.cache = (X, weights, bias)
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

    #HORRRIBLE RUNNING TIME, ABSOLUTELY HORRIBLE    
    def conv_forward(self, X, weights, bias, pad = 0, stride = 1):
        """
         X: (m, channels, heights, widths)
         weights (bunch of filters params): (n,channels, height, width)
        """
        m, c, h, w = X.shape
        n_fil, _, h_fil, w_fil = weights.shape

        print("X:",X.shape)
        print("filter:", weights.shape)
        #output dims
        out_h = int((h-h_fil+2*pad)/stride + 1) 
        out_w = int((w-w_fil+2*pad)/stride + 1)

        out = np.zeros((m, n_fil, out_h, out_w))
        print("Out:",out.shape)
        for l in range(m):
            #padding
            pad_dims = [(0,0),[pad,pad],[pad,pad]]
            img = np.pad(X[l,:], pad_dims, 'constant', constant_values=(0,0))
            if(l%200 == 0): print("forwardProb current on element:{}".format(str(l)))
            for k in range(n_fil): #depths
                for j in range(out_h): #height
                    for i in range(out_w): #width
                        img_parse = img[:,stride*j:h_fil+ stride*j, stride*i: w_fil+ stride*i]
                        out[l,k,j,i] = np.sum(img_parse*weights[k, :, :, :]) + bias[k]

        self.cache = (X, weights, bias, pad, stride)
        return (out, self.cache)

    def conv_vec_forward(self, x, w, b, pad = 0, stride = 1):
      """
      By Standford cs231n
      A fast implementation of the forward pass for a convolutional layer
      based on im2col and col2im.
      """
      N, C, H, W = x.shape
      num_filters, _, filter_height, filter_width = w.shape

      # Check dimensions
      assert (W + 2 * pad - filter_width) % stride == 0, 'width does not work'
      assert (H + 2 * pad - filter_height) % stride == 0, 'height does not work'

      # Create output
      out_height = int((H + 2 * pad - filter_height) / stride + 1)
      out_width = int((W + 2 * pad - filter_width) / stride + 1)
      out = np.zeros((N, num_filters, out_height, out_width), dtype=x.dtype)

      # x_cols = im2col_indices(x, w.shape[2], w.shape[3], pad, stride)
      x_cols = im2col_cython(x, w.shape[2], w.shape[3], pad, stride)
      res = w.reshape((w.shape[0], -1)).dot(x_cols) + b.reshape(-1, 1)

      out = res.reshape(w.shape[0], out.shape[2], out.shape[3], x.shape[0])
      out = out.transpose(3, 0, 1, 2)

      cache = (x, w, b, pad, stride, x_cols)
      return out, cache
    
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

    def conv_vec_backward(self, dout, cache):
      """
      By standford cs231n 
      A fast implementation of the backward pass for a convolutional layer
      based on im2col and col2im.
      """
      x, w, b, pad, stride, x_cols = cache

      db = np.sum(dout, axis=(0, 2, 3))

      num_filters, _, filter_height, filter_width = w.shape
      dout_reshaped = dout.transpose(1, 2, 3, 0).reshape(num_filters, -1)
      dw = dout_reshaped.dot(x_cols.T).reshape(w.shape)

      dx_cols = w.reshape(num_filters, -1).T.dot(dout_reshaped)
      # dx = col2im_indices(dx_cols, x.shape, filter_height, filter_width, pad, stride)
      dx = col2im_cython(dx_cols, x.shape[0], x.shape[1], x.shape[2], x.shape[3],
                         filter_height, filter_width, pad, stride)

      return dx, dw, db
    
    def max_pooling_forward(self, X, poolH, poolW, stride, *args):
        """
           max pooling forward pass.
        """
        m, c, h, w = X.shape

        #output dims
        out_h = int((h-poolH)/stride + 1)
        out_w = int((w-poolW)/stride + 1)
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
        self.cache = X
        return (self.activFunc.getVal(X), self.cache)
    
    def ReLU_backward(self, X, cache, *args):
        """ ReLU layer backward """
        return ActivationFunctions("ReLU").getVal(X, True, cache)

    def LReLU_forward(self, X, *args):
        """ Leaky ReLU layer forward"""
        self.activFunc = ActivationFunctions("LReLU")
        self.cache = X
        return self.activFunc.getVal(X), self.cache
    
    def LReLU_backward(self, X, cache, *args):
        """ Leaky ReLU layer backward """
        return ActivationFunctions("LReLU").getVal(X, True, cache)
    
    def affine_ReLU_forward(self, X = None, weights = None, *args):
        """ combining both fully connected and ReLU """

        z1, cache_af= self.affine_forward(X, weights)
        a1, cache_ReLU= self.ReLU_forward(z1)
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
        a1, cache_LReLU= self.ReLU_forward(z1)
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
        self.cache = X
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
