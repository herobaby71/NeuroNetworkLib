import numpy as np
class ConvLayer(object):
    def __init__(self, )
    
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
