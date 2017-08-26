import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import optimize
from sklearn.model_selection import train_test_split
from random import shuffle
from FeedForwardNeuroNet import MultiLayerNeuroNet

train_data = pd.read_csv('train.csv', sep = ',')
test_data = pd.read_csv('test.csv', sep = ',')

#normalize the inputs value from 0-> 255 to 0-> 1
def normalize(X):
    return X/255

#label in column 0
#pixels of the image in column 0 to 783
train_data = np.array(train_data)
y = train_data[:,0]
X = normalize(train_data[:,1:785])

#convert y into a matrix
y_matrix = np.zeros([X.shape[0], 10])
print("y-matrix shape:", y_matrix.shape)
for i in range(X.shape[0]):
    y_matrix[i][y[i]] = 1

#visualize 100 random digits
##randno = X[np.random.choice(X.shape[0],100, replace=False), :]
##image = np
##display_array = None
##display_row = np.reshape(randno[0],(28,28))
##count = 1
##while(count<100):
##    #concatinate to the right
##    while(not(count%10 ==0)):
##        display_row = np.concatenate((display_row, np.reshape(randno[count], (28,28))), axis = 1)
##        count+=1
##    if(display_array is None):
##        display_array = display_row
##    else:
##        display_array= np.concatenate((display_array, display_row), axis = 0)
##    if (count == 100):
##        break
##    display_row = np.reshape(randno[count],(28,28))
##    count+=1
##fig = plt.imshow(display_array)
##fig.axes.get_xaxis().set_visible(False)
##fig.axes.get_yaxis().set_visible(False)
##fig.set_cmap("gray")
##plt.show(block=False)


#Model

##Model2 = MultiLayerNeuroNet(X, y_matrix, 784, 397, 10, "sigmoid", .01)
##Model3 = MultiLayerNeuroNet(X, y_matrix, 784, 397, 10, "sigmoid", 0)
##Model4 = MultiLayerNeuroNet(X, y_matrix, 784, 397, 10, "sigmoid", .2)
##Model5 = MultiLayerNeuroNet(X, y_matrix, 784, 397, 10, "sigmoid", .5)
##Model6 = MultiLayerNeuroNet(X, y_matrix, 784, 397, 10, "sigmoid", 2)
Model7 = MultiLayerNeuroNet(X, y_matrix, 784, 529, 10, .2, "sigmoid")


# In[114]:

print("Model 7")
Model7.train()
print(Model2.accuracy(ModelTemp.X_train,ModelTemp.y_train))
print(Model2.accuracy(ModelTemp.X_cv,ModelTemp.y_cv))


