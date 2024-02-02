import copy
import matplotlib.pyplot as plt
from neuralnet import *

def train(model, x_train, y_train, x_valid, y_valid, config):
    epoch, batchSize = config['epochs'], config['batch_size']
    early, stop_point = config['early_stop'], config['early_stop_epoch']
    prev_loss, patience = 0, 0

    epoch_best, stopped = epoch, False
    valid_loss_pts, train_loss_pts = [], []
    valid_acc_pts, train_acc_pts = [], []
    train_size, valid_size = len(x_train), len(x_valid)
    for e in range(1, epoch + 1):
        # Begin batch learning and collect the training set's loss.
        train_acc, train_loss = 0, 0
        x_train_n = x_train.shape[0]
        for i in range(0, x_train_n, batchSize):
            x_train_batch, y_train_batch  = x_train[i:i+batchSize,:], y_train[i:i+batchSize,:]
            data = model.forward(x_train_batch, y_train_batch)
            train_acc += data[0]
            train_loss += data[1]
            model.backward()

        # Calculate accurarcy and normalize the loss by set size.
        valid_acc, valid_loss = model.forward(x_valid, y_valid)
        valid_acc /= valid_size
        train_acc /= train_size
        valid_loss /= valid_size
        train_loss /= train_size

        # Save the points for a plot.
        valid_loss_pts.append(valid_loss)
        train_loss_pts.append(train_loss)
        valid_acc_pts.append(valid_acc)
        train_acc_pts.append(train_acc)
        if not stopped: print("Epoch: %d | Test Set (Loss: %.10f) | Validation Set (Accuracy: %.10f, Loss: %.10f)" % (e, train_loss, valid_acc, valid_loss))

        # Detect when to stop training
        if prev_loss - valid_loss < 0: patience += 1
        else: patience = 0
        if early and patience == stop_point:
            epoch_best, stopped, early = e - stop_point, True, False
            print("Early Stoppage at Epoch: %d, Best Weights Are at Epoch: %d" % (e, epoch_best))
            break
        prev_loss = valid_loss
    
    # Plot the train/valid loss/accuracy
    # myPlots(train_loss_pts, train_acc_pts, valid_loss_pts, valid_acc_pts, epoch_best)
    return model

def myPlots(train_loss_pts, train_acc_pts, valid_loss_pts, valid_acc_pts, epoch_best):
    # Loss Plot
    plt.plot(train_loss_pts, label='Train Set')
    plt.plot(valid_loss_pts, label='Valid Set')
    plt.vlines(epoch_best, min(train_loss_pts), valid_loss_pts[epoch_best - 1], linestyle='dashed', label=f'Best Epoch: {epoch_best}')
    plt.title("Multi-Layer Sigmoid Activation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    # Accuracy Plot
    plt.plot(train_acc_pts, label="Train Set")
    plt.plot(valid_acc_pts, label="Valid Set")
    plt.vlines(epoch_best, min(train_acc_pts), valid_acc_pts[epoch_best - 1], linestyle='dashed', label=f'Best Epoch: {epoch_best}')
    plt.title("Multi-Layer Sigmoid Activation")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()

def modelTest(model, X_test, y_test):
    acc, loss = model.forward(X_test, y_test)
    acc /= X_test.shape[0]
    loss /= X_test.shape[0]
    return [acc, loss]