import numpy as np
import pandas as pd
def openThetas(file_name):
    Thetas = np.load(file_name)
    T1,T2 = unrollThetaParams(Thetas, 784,784,10)
    weights = {"W0": T1, "W1": T2}
    return weights

def openMNISTData():
    train_data = pd.read_csv('train.csv', sep = ',')
    test_data = pd.read_csv('test.csv', sep = ',')
    
    train_data = np.array(train_data)
    y = train_data[:,0]
    X = train_data[:,1:785]/255
    y_matrix = np.zeros([X.shape[0], 10])
    for i in range(X.shape[0]):
        y_matrix[i][y[i]] = 1

    return (X,y_matrix)

def unrollThetaParams(Thetas = None, input_layer_size=0,hidden_layer_size = 0, output_layer_size = 0):
    if(Thetas is None):
        return
    Theta0 = np.reshape(Thetas[0:hidden_layer_size*(input_layer_size+1)],
                        (hidden_layer_size,input_layer_size+1))
    Theta1 = np.reshape(Thetas[hidden_layer_size*(input_layer_size+1):],
                        (output_layer_size,hidden_layer_size+1))
    return (Theta0, Theta1)

def InitializeWeights(input_layer_size, out_layer_size):        
    return np.sqrt(2/(input_layer_size+out_layer_size))*np.random.randn(out_layer_size,input_layer_size+1)

def predict(costFunction,model, X):
    a2 = costFunction(model, X, None, False)
    return np.argmax(a2, axis=1)

def accuracy(costFunction, model, X, y):
    pred = predict(costFunction,model, X)
    target = np.argmax(y, axis = 1)
    return np.mean(list(map(int,np.equal(pred,target))))
