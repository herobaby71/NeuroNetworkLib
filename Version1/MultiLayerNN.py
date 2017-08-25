import numpy as np
import pandas as pd
import cv2
from collections import Counter
from scipy import optimize
from random import shuffle
from sklearn.model_selection import train_test_split
from scipy.special import expit

#process raw data by balancing it, shuffle and split into train/test/cv
def preprocessing(X,y):
    
    #train: 70%, CV: 15%, test: 15%
    X_train, X_test, y_train, y_test = train_test_split(
       X, y, test_size = .3)
    #split the test set to CV and test
    X_test,X_cv, y_test, y_cv = train_test_split(
        X, y, test_size = .5)

    return (X_train, y_train, X_cv,y_cv, X_test,y_test)
    
#softmax activation function
def sigmoid(z):
    return expit(z)
def softmax(z):
    """return the softmax values"""
    return np.exp(z) / np.sum(np.exp(z), axis=0)
#derivative of the sigmoid
def gradientSigmoid(z):
    return sigmoid(z)*(1-sigmoid(z))

#A very straightforward 3 layer neuronet with logSigmoidal(i tried linear does not work as well) neurons
class modelA():
    def __init__(self,X,y, input_layer_size = None, hidden_layer_size = 200, output_layer_size = None):
        #m: total number of data
        self.m = X.shape[0]
        self.X = X
        self.y = y
        self.iter = 1
        self.X_train, self.y_train, self.X_cv, self.y_cv, self.X_test, self.y_test = preprocessing(X,y)
#np.concatenate((Thood0_48[:, None], h_labels), axis=1)
        #layer sizes
        self.input_layer_size = (self.X[0].shape[0]*self.X[0].shape[1]) if (input_layer_size is None) else input_layer_size
        self.hidden_layer_size = hidden_layer_size
        self.output_layer_size = output_layer_size

        #Weights/Thetas w/ bias
        Init_Theta1 = np.random.random((hidden_layer_size, input_layer_size + 1))*2-1
        Init_Theta2 = np.random.random((output_layer_size, hidden_layer_size + 1))*2-1
##        Init_Theta1 = np.zeros((hidden_layer_size, input_layer_size + 1))
##        Init_Theta2 = np.zeros((output_layer_size, hidden_layer_size + 1))       
        self.Thetas = np.append(Init_Theta1.ravel(),Init_Theta2.ravel())

        
    def visualizeData(self):
        for i in range(0, self.m):
                                                                        #cv2 cant show image if the datatype is different
            cv2.imshow("img", np.array(np.reshape(self.X[i], (48,64)),dtype='uint8'))
            if cv2.waitKey(25) & 0xFF==ord('q'):
                cv2.destroyAllWindows()
                break

    def unrollParams(self, Thetas = None):
        if(Thetas is None):
            Thetas = self.Thetas
        Theta1 = np.reshape(Thetas[0:self.hidden_layer_size*(self.input_layer_size+1)], (self.hidden_layer_size, (self.input_layer_size+1)))
        Theta2 = np.reshape(Thetas[self.hidden_layer_size*(self.input_layer_size+1):], (self.output_layer_size, self.hidden_layer_size+1))
        return (Theta1, Theta2)
    
    def predict(self, X):
        """predict the outcome either the cross-validation or the test set depending on the input name"""
        m = X.shape[0]
        Theta1,Theta2 = self.unrollParams()
        pred = np.array([m,1])
        h1 = sigmoid(np.dot(np.append(np.ones([m,1]), X, axis = 1), Theta1.T))
        h2 = sigmoid(np.dot(np.append(np.ones([m,1]), h1, axis = 1), Theta2.T))
        pred = np.argmax(h2, axis = 1)
        return pred 
    def getAccuracy(self, X, y):
        pred = self.predict(X)
        target = np.argmax(y, axis = 1)
        return np.mean(list(map(int,np.equal(pred,target))))
    def forwardProp(self, Thetas):
        X = self.X
        y = self.y
        m = X.shape[0]
        Theta1,Theta2 = self.unrollParams(Thetas)
        a0 = np.append(np.ones([m,1]), X, axis = 1)
        #compute the logits and apply softmax function to them on each layer
        h1 = np.dot(a0, Theta1.T)
        a1 = softmax(h1)
        h2 = np.dot(np.append(np.ones([m,1]), h1, axis = 1), Theta2.T)
        a2 = softmax(h2)
        return a2
    
    def CostFunction(self, Thetas, lamb = 1):
        """
           Compute the corss entropy error (which we want to minimize),
           and the gradient using BackPropagation algorithm
        """
        X = self.X_train
        y = self.y_train
        m = X.shape[0]
        Theta1,Theta2 = self.unrollParams(Thetas)
        
        #forward propagation
        #add bias
        a0= np.append(np.ones([m,1]),X,axis = 1)
        a1 = sigmoid(np.dot(a0,Theta1.T))
        a1 = np.append(np.ones([m,1]), a1, axis = 1)
        a2 = sigmoid(np.dot(a1,Theta2.T))
        #calculate the cross entropy cost error with regularization 
        J = -(1/m)*(np.sum(y*np.log(a2) + (1-y)*np.log(1-a2)))
        
        #Regularization. no regularization on the bias term, which is in the column 0 of Thetas
        #J+= lamb/(2*m) * (np.sum(np.power(Theta1[:,1:],2)) + np.sum(np.power(Theta2[:,1:],2)))

        print("Cost after {} iterations:".format(str(self.iter)), J)
        self.iter+=1
        #Backpropagation algorithm
        Delta1 = np.zeros(Theta1.shape)
        Delta2 = np.zeros(Theta2.shape)
        for i in range(m):
            a1 = a0[i]
            a2 = sigmoid(np.dot(a1,Theta1.T))
            a2 = np.concatenate((np.array([1]), a2))
            a3 = sigmoid(np.dot(a2,Theta2.T))
            delta3 = a3.ravel()-y[i]
            delta2 = (np.dot(Theta2[:,1:].T,delta3).T)* gradientSigmoid(np.dot(a1,Theta1.T))

            #numpy made it easy. outer calculate it correctly {[a,b,c]]* [[1],[2],[3]]
            Delta1 = Delta1 + np.outer(delta2,a1)
            Delta2 = Delta2 + np.outer(delta3,a2)
        
        Grad2 = (1/m)*Delta2
        Grad2[:,2:] = Grad2[:,2:] + (lamb/m)*Theta2[:,2:]
        Grad1 = (1/m)*Delta1
        Grad1[:,2:] = Grad1[:,2:] + (lamb/m)*Theta1[:,2:]

        Grad = np.append(Grad1.ravel(),Grad2.ravel())

        return [J, Grad]

    def train(self, lamb = 1):
        """ given the regularization parameter lambda, train the model using advance optimization BFGS"""
        arguments = (lamb)
        print("training...")
        results = optimize.minimize(self.CostFunction,x0 = self.Thetas, args = arguments, options = {'disp':True, 'maxiter': 40}, method = "L-BFGS-B", jac = True)
        self.Thetas = results['x']
        print("successfully trained the model")
        

file_name = "training.npy"
train_data = np.load(file_name)

#{0:"forward", 1:"forward_left", 2:"forward_right", 3:"reverse",
#   4:"reverse_left", 5:"reverse_right", 6:"idle", 7:"right", 8:"left"}
    
#print(Counter(df[1].apply(str)))
#Counter({'1': 1039, '6': 680, '0': 387, '3': 231,'8': 184, '2': 93, '4': 83, '7': 3})
    
#For the sake of simplicity, let's get rid of commands 4,5,7,8.
#and command "idle" which has value of 6 will now be in indice 4
DataX = np.empty((2430,3072))
DataY = np.zeros((2430,5))
count = 0
for i in range(train_data.shape[0]):
    data = train_data[i]
    img = np.reshape(cv2.cvtColor(data[0], cv2.COLOR_BGR2GRAY), (1,3072))
    key = int(data[1])
    if(key == 6 or key == 0 or key == 1 or key == 2 or key == 3):
        DataX[count] = img
        if(key == 6):
            DataY[count][4] = 1
        else:
            DataY[count][key] = 1
        count+=1
X = np.array(pd.read_csv('data_X.csv', sep = ',', header = None))
y = np.array(pd.read_csv('data_y.csv', sep = ',', header = None))
y_matrix = np.zeros((len(y), 10))
for i in range(len(y)):
    y_matrix[i][y[i]-1] = 1


A = modelA(DataX,DataY, 3072,200,5)
C = modelA(X,y_matrix, 400,25,10)
def main():
    return
if __name__ == "__main__":
    main()



















    
