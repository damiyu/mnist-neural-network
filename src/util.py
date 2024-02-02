import os, gzip
import struct
import yaml
import numpy as np
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import constants
from urllib.request import urlretrieve

def load_config(path): return yaml.load(open(path, 'r'), Loader=yaml.SafeLoader)
def normalize_data(inp):
    # Mean and STD of the first three images
    # for i in range(3): print(f"Image {i + 1} | Mean: {np.mean(inp[i])} | SD: {np.std(inp[i])}")
    return (inp - np.mean(inp, axis=1)[:,np.newaxis]) / np.std(inp, axis=1)[:,np.newaxis]

def one_hot_encoding(labels, num_classes=10):
    n = len(labels)
    encode = [[0 for _ in range(num_classes)] for _ in range(n)]

    for i in range(n): encode[i][int(labels[i][0])] = 1
    return np.array(encode)

def plots(trainEpochLoss, trainEpochAccuracy, valEpochLoss, valEpochAccuracy, earlyStop):
    if not os.path.exists(constants.saveLocation): os.makedirs(constants.saveLocation)

    fig1, ax1 = plt.subplots(figsize=((24, 12)))
    epochs = np.arange(1,len(trainEpochLoss)+1,1)
    ax1.plot(epochs, trainEpochLoss, 'r', label="Training Loss")
    ax1.plot(epochs, valEpochLoss, 'g', label="Validation Loss")
    plt.scatter(epochs[earlyStop],valEpochLoss[earlyStop],marker='x', c='g',s=400,label='Early Stop Epoch')
    plt.xticks(ticks=np.arange(min(epochs),max(epochs)+1,10), fontsize=35 )
    plt.yticks(fontsize=35)
    ax1.set_title('Loss Plots', fontsize=35.0)
    ax1.set_xlabel('Epochs', fontsize=35.0)
    ax1.set_ylabel('Cross Entropy Loss', fontsize=35.0)
    ax1.legend(loc="upper right", fontsize=35.0)
    plt.savefig(constants.saveLocation+"loss.eps")
    plt.show()

    fig2, ax2 = plt.subplots(figsize=((24, 12)))
    ax2.plot(epochs, trainEpochAccuracy, 'r', label="Training Accuracy")
    ax2.plot(epochs, valEpochAccuracy, 'g', label="Validation Accuracy")
    plt.scatter(epochs[earlyStop], valEpochAccuracy[earlyStop], marker='x', c='g', s=400, label='Early Stop Epoch')
    plt.xticks(ticks=np.arange(min(epochs),max(epochs)+1,10), fontsize=35)
    plt.yticks(fontsize=35)
    ax2.set_title('Accuracy Plots', fontsize=35.0)
    ax2.set_xlabel('Epochs', fontsize=35.0)
    ax2.set_ylabel('Accuracy', fontsize=35.0)
    ax2.legend(loc="lower right", fontsize=35.0)
    plt.savefig(constants.saveLocation+"accuarcy.eps")
    plt.show()

    #Saving the losses and accuracies for further offline use
    pd.DataFrame(trainEpochLoss).to_csv(constants.saveLocation+"trainEpochLoss.csv")
    pd.DataFrame(valEpochLoss).to_csv(constants.saveLocation+"valEpochLoss.csv")
    pd.DataFrame(trainEpochAccuracy).to_csv(constants.saveLocation+"trainEpochAccuracy.csv")
    pd.DataFrame(valEpochAccuracy).to_csv(constants.saveLocation+"valEpochAccuracy.csv")


def createTrainValSplit(x_train,y_train):
    # Zip the data together before shuffling.
    n = len(x_train)
    pair = [[x_train[i], y_train[i]] for i in range(n)]
    np.random.seed(69)
    np.random.shuffle(pair)

    # Return a 80/20 train/valid split of the data.
    train = [np.array([pair[i][0] for i in range(int(0.8 * n))]), np.array([pair[i][1] for i in range(int(0.8 * n))])]
    valid = [np.array([pair[i][0] for i in range(int(0.8 * n), n)]), np.array([pair[i][1] for i in range(int(0.8 * n), n)])]
    return [np.array(train[0]), np.array(train[1]), np.array(valid[0]), np.array(valid[1])]


def get_mnist():
    def load_data(src, num_samples):
        gzfname, h = urlretrieve(src, "./delete.me")
        try:
            with gzip.open(gzfname) as gz:
                n = struct.unpack("I", gz.read(4))
                # Read magic number.
                if n[0] != 0x3080000:
                    raise Exception("Invalid file: unexpected magic number.")
                # Read number of entries.
                n = struct.unpack(">I", gz.read(4))[0]
                if n != num_samples:
                    raise Exception(
                        "Invalid file: expected {0} entries.".format(num_samples)
                    )
                crow = struct.unpack(">I", gz.read(4))[0]
                ccol = struct.unpack(">I", gz.read(4))[0]
                if crow != 28 or ccol != 28:
                    raise Exception(
                        "Invalid file: expected 28 rows/cols per image."
                    )
                # Read data.
                res = np.frombuffer(
                    gz.read(num_samples * crow * ccol), dtype=np.uint8
                )
        finally:
            os.remove(gzfname)
        return res.reshape((num_samples, crow, ccol)) / 256

    def load_labels(src, num_samples):
        gzfname, h = urlretrieve(src, "./delete.me")
        try:
            with gzip.open(gzfname) as gz:
                n = struct.unpack("I", gz.read(4))
                # Read magic number.
                if n[0] != 0x1080000:
                    raise Exception("Invalid file: unexpected magic number.")
                # Read number of entries.
                n = struct.unpack(">I", gz.read(4))
                if n[0] != num_samples:
                    raise Exception(
                        "Invalid file: expected {0} rows.".format(num_samples)
                    )
                # Read labels.
                res = np.frombuffer(gz.read(num_samples), dtype=np.uint8)
        finally:
            os.remove(gzfname)
        return res.reshape((num_samples))

    def try_download(data_source, label_source, num_samples):
        data = load_data(data_source, num_samples)
        labels = load_labels(label_source, num_samples)
        return data, labels

    server = 'https://raw.githubusercontent.com/fgnt/mnist/master'

    # URLs for the train image and label data
    url_train_image = f'{server}/train-images-idx3-ubyte.gz'
    url_train_labels = f'{server}/train-labels-idx1-ubyte.gz'
    num_train_samples = 60000

    train_features, train_labels = try_download(url_train_image, url_train_labels, num_train_samples)

    # URLs for the test image and label data
    url_test_image = f'{server}/t10k-images-idx3-ubyte.gz'
    url_test_labels = f'{server}/t10k-labels-idx1-ubyte.gz'
    num_test_samples = 10000

    test_features, test_labels = try_download(url_test_image, url_test_labels, num_test_samples)

    return train_features, train_labels, test_features, test_labels


def load_data(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print('Fetching MNIST data...')
        train_features, train_labels, test_features, test_labels = get_mnist()
        # Save data using pickle
        with open(os.path.join(path, "mnist.pkl"), "wb") as f:
            pickle.dump([train_features, train_labels, test_features, test_labels], f)
        print(f'Done. All data can be found in {path}')

    # Load data from pickle file
    print(f'Loading MNIST data from {path}mnist.pkl')
    with open(f"{path}mnist.pkl", 'rb') as f:
        train_images, train_labels, test_images, test_labels = pickle.load(f)
    print('Done.\n')

    # Reformat the images and labels
    train_images, test_images = train_images.reshape(train_images.shape[0], -1), test_images.reshape(test_images.shape[0], -1)
    train_labels, test_labels = np.expand_dims(train_labels, axis=1), np.expand_dims(test_labels, axis=1)

    # Create 80-20 train-validation split
    train_images, train_labels, val_images, val_labels = createTrainValSplit(train_images, train_labels)

    # Preprocess data
    train_normalized_images = normalize_data(train_images)
    train_one_hot_labels = one_hot_encoding(train_labels, num_classes=10)  # (n, 10)

    # Three examples of what each image looks like after normalization, uncomment to see.
    '''
    img1 = np.array([train_normalized_images[0]]).reshape(28, 28)
    img2 = np.array([train_normalized_images[1]]).reshape(28, 28)
    img3 = np.array([train_normalized_images[2]]).reshape(28, 28)
    plt.imshow(img1, interpolation='nearest')
    plt.show()
    plt.imshow(img2, interpolation='nearest')
    plt.show()
    plt.imshow(img3, interpolation='nearest')
    plt.show()
    '''

    val_normalized_images = normalize_data(val_images)
    val_one_hot_labels = one_hot_encoding(val_labels, num_classes=10)  # (n, 10)

    test_normalized_images = normalize_data(test_images)
    test_one_hot_labels = one_hot_encoding(test_labels, num_classes=10)  # (n, 10)

    # Get set sizes.
    # print(f"Train: {len(train_images)} | Valid: {len(val_images)} | Test: {len(test_images)}")
    return train_normalized_images, train_one_hot_labels, val_normalized_images, val_one_hot_labels, test_normalized_images, test_one_hot_labels