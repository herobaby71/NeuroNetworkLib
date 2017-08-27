import numpy as np
import pandas as pd
import pickle
def openThetas(file_name):
    """ Upgrade to be able to open any number of params"""
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

def openCIFAR10Data(path = "cifar-10-batches-py/"):
    file_name = path + "data_batch_"
    #50000 images of size 32x32x3
    train_data_set = np.zeros([50000, 3,32, 32], dtype = float)
    train_labels = np.zeros([50000, 1], dtype= int)
    for i in range(1,6):
        file = open(file_name + str(i), 'rb')
        raw_data = pickle.load(file, encoding='bytes')


        batch_labels = np.reshape(np.array(raw_data[b'labels']), (10000, 1))
        batch_imgs= np.array(raw_data[b'data'], dtype = float) / 255 #normalize
        batch_imgs = np.reshape(batch_imgs, [-1,3,32,32])

        #append
        m = len(batch_imgs)
        train_data_set[m*(i-1):m*i, :] = batch_imgs
        train_labels[m*(i-1):m*i,:] = batch_labels
        
        file.close()

    label_matrix = np.zeros([train_labels.shape[0], 10])
    for i in range(train_labels.shape[0]):
        label_matrix[i][train_labels[i][0]] = 1    

    labels_dict = {0: "airplane", 1: "automobile", 2: "bird", 3: "cat", 4: "deer", 5: "dog", 6: "frog", 7: "horse", 8: "ship", 9:"truck"}
    return (train_data_set, label_matrix, labels_dict)

def openCIFAR100Data():
    pass

def unrollThetaParams(Thetas = None, input_layer_size=0,hidden_layer_size = 0, output_layer_size = 0):
    if(Thetas is None):
        return
    Theta0 = np.reshape(Thetas[0:hidden_layer_size*(input_layer_size+1)],
                        (hidden_layer_size,input_layer_size+1))
    Theta1 = np.reshape(Thetas[hidden_layer_size*(input_layer_size+1):],
                        (output_layer_size,hidden_layer_size+1))
    return (Theta0, Theta1)

def InitializeConvWeights(n_filters= 32, filter_dims = (3,5,5)):
    W = 1e-3* np.random.randn(n_filters, filter_dims[0], filter_dims[1], filter_dims[2])
    b = 0*np.random.randn(n_filters)
    return (W,b)
def InitializeFullConnectWeights(input_layer_size, out_layer_size):        
    return np.sqrt(2/(input_layer_size+out_layer_size))*np.random.randn(out_layer_size,input_layer_size+1)

def predict(costFunction,model, X):
    a2 = costFunction(model, X, None, False)
    return np.argmax(a2, axis=1)

def accuracy(costFunction, model, X, y):
    pred = predict(costFunction,model, X)
    target = np.argmax(y, axis = 1)
    return np.mean(list(map(int,np.equal(pred,target))))
