import numpy as np
from usefulFuncs import predict, accuracy
class Optimizer(object):
    def __init__(self, update_method = 'sgd'):
        #if we use momentum update, we need to store velocities value
        self.step_cache = {}    

    def l_bfgs(self, X,y ): pass
    def train(self, X, y, X_val, y_val,
              model, costFunction,
              learningRate = 1e-2, momentum = 0, learningRateDecay = .95,
              update = 'sgd',sampleBatches = True,
              numEpochs = 30, batchSize = 1000, accFreq = None,
              verbose =True):

        m = X.shape[0]     
        
        #check if we are going to use minibatch or not
        iterationsPerIter = 1
        if(sampleBatches):
            iterationsPerIter = int(m/batchSize)

        numIters = numEpochs*iterationsPerIter
        epoch = 0
        best_val_acc = 0.0
        best_model = {}
        loss_history = []
        train_acc_history = []
        val_acc_history = []

        for it in range(numIters):
            if(it%10 ==0): print("starting iteration {}".format(str(it)))

            if(sampleBatches):
                #pick batchSize random values
                batch_mask = np.random.choice(m, batchSize)
                X_batch = X[batch_mask]
                y_batch = y[batch_mask]
            else:
                X_batch = X
                y_batch = y

            #evaluate cost and gradient
            cost,grads = costFunction(model, X_batch,y_batch)
            loss_history.append(cost)

            #param update
            for p in model:
                if(update == 'sgd'):
                    dx = -learningRate*grads[p]
                elif(update == 'momentum'):
                    if(not(p in self.set_cache)):
                        self.step_cache[p] = np.zeros(grads[p].shape)
                    #momentum update:
                    dx = momentum*self.step_cache[p] - learningRate*grads[p]
                    self.step_cache[p] = dx

                elif(update == 'rmsprop'):
                    decay_rate = .99
                    if(not(p in self.step_cache)):
                        self.step_cache[p] = np.zeros(grads[p].shape)

                    #RMSProp
                    self.step_cache[p] = decay_rate* self.step_cache[p] + (1-decay_rate)* grads[p]**2
                    dx = -learningRate * grads[p] / np.sqrt(self.step_cache[p] + 1e-8)
                else:
                    raise ValueError('Unrecognized update type "%s"' % update)

                model[p]+=dx #update params

            # every epoch perform an evaluation on the validation set
            first_it = (it == 0)
            epoch_end = (it + 1) % iterationsPerIter == 0
            acc_check = (accFreq is not None and it % accFreq == 0)
            if first_it or epoch_end or acc_check:
                if it > 0 and epoch_end:
                    # decay the learning rate
                    learningRate *= learningRateDecay
                    epoch += 1

            # evaluate train accuracy
            train_mask = np.random.choice(m, 1000)
            X_train_subset = X[train_mask] if m > 1000 else X
            y_train_subset = y[train_mask] if m > 1000 else y

            # evaluate train accuracy
            train_acc = accuracy(costFunction, model, X_train_subset, y_train_subset)
            train_acc_history.append(train_acc)

            # evaluate val accuracy
            val_acc = accuracy(costFunction, model, X_val, y_val)
            val_acc_history.append(val_acc)
            
            # keep track of the best model based on validation accuracy
            if val_acc > best_val_acc:
              # make a copy of the model
              best_val_acc = val_acc
              best_model = {}
              for p in model:
                best_model[p] = model[p].copy()

            # print progress if needed
            if verbose:
              print ('Finished epoch %d / %d: cost %f, train: %f, val %f, lr %e'
                     % (epoch, numEpochs, cost, train_acc, val_acc, learningRate))

        if verbose:
          print ('finished optimization. best validation accuracy: %f' % (best_val_acc, ))
        # return the best model and the training history statistics
        return best_model, loss_history, train_acc_history, val_acc_history

















