# NeuroNetworkLib
This lib is inspired by Andrew Ng's coursera class and Stanford's cs231n.

Version 1) a 3-layers fully connected neurons with regularization using sklearn l-bfgs second order optimization method.

Version 2) added first-order optimization (revision from cs231n).

Version 3) added more activation functions, added convolutional layers and max-pooling layers. Able to create a generalize network as an object (Architecture) and call addLayer() methods. Call .train() to train the model.

Drawbacks) the implementations of convolutional layers are in python (not C) + unvectorized, thus the runtime is ridiculously slow. This is more of a learning project than a applicable one. => Tensorflow time.
