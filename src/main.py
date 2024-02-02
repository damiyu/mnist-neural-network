import gradient
from constants import *
from train import *
from gradient import *
import argparse

def main(args):
    configFile = None
    if (args.experiment == 'test_softmax'): configFile = "softmax.yaml"
    elif (args.experiment == 'test_gradients'): configFile = "gradient.yaml"
    elif (args.experiment == 'test_momentum'): configFile = "momentum.yaml"
    elif (args.experiment == 'test_regularization_1'): configFile = "regularization1.yaml"
    elif (args.experiment == 'test_regularization_2'): configFile = "regularization2.yaml"
    elif (args.experiment == 'test_regularization_3'): configFile = "regularization3.yaml"
    elif (args.experiment == 'test_relu'): configFile = "relu.yaml"
    elif (args.experiment == 'test_sigmoid'): configFile = "sigmoid.yaml"

    # Load the data
    x_train, y_train, x_valid, y_valid, x_test, y_test = util.load_data(path=datasetDir)

    # Load the configuration from the corresponding yaml file. Specify the file path and name
    config = util.load_config(configYamlPath + configFile)

    if(args.experiment == 'test_gradients'):
        gradient.checkGradient(x_train, y_train, config)
        return 1

    # Create a Neural Network object which will be our model
    model = Neuralnetwork(config)

    # train the model
    model = train(model, x_train, y_train, x_valid, y_valid, config)

    # test the model
    test_acc, test_loss = modelTest(model, x_test, y_test)

    #Print test accuracy and test loss
    print('Test Accuracy:', test_acc, ' Test Loss:', test_loss)


if __name__ == "__main__":
    #Parse the input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', type=str, default='test_momentum', help='Specify the experiment that you want to run')
    args = parser.parse_args()
    main(args)