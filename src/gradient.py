import numpy as np
from neuralnet import Neuralnetwork

def check_grad(model, x_train, y_train):
    # My e = 10 ^ -2 and my gradient should not exceed O(e^2).
    n, epsilon = len(x_train), 10 ** -2
    big_o = epsilon ** 2

    # List of which weights to modify by e with their locations.
    locations = [("output", 129, 10, 128, 0), ("hidden", 785, 128, 784, 0), ("output", 129, 10, 0, 0),
                 ("output", 129, 10, 0, 1), ("hidden", 785, 128, 0, 0), ("hidden", 785, 128, 0, 1)]

    # Perform a gradient check on every pattern on the small section of the train set.
    for i in range(n):
        if not locations: break

        patt, expect = x_train[i:i+1,:], y_train[i:i+1,:]
        # First obtain the difference loss of the pattern.
        modify_location = locations.pop()
        acc, pos_e_loss = model.forward(patt, expect, True, modify_location, epsilon)
        # I am modifying the specific weight by -2e because I need to "undo" the original +e.
        acc, neg_e_loss = model.forward(patt, expect, True, modify_location, -2 * epsilon)

        # Get the approximation the slope of the specific weight.
        slope = abs(pos_e_loss - neg_e_loss) / (2 * epsilon)
        model.forward(patt, expect, True, modify_location, epsilon)
        grad = abs(model.backward(True, modify_location))
        diff = abs(slope - grad)
        passed = diff <= big_o

        # Compare approximation vs backpropagation
        print(f"Pattern {i + 1} ({passed}): | {modify_location} | Slope ({slope} | Back ({grad}) | Diff ({diff})")


def checkGradient(x_train, y_train, config):
    subsetSize = 10  # Feel free to change this
    sample_idx = np.random.randint(0, len(x_train), subsetSize)
    x_train_sample, y_train_sample = x_train[sample_idx], y_train[sample_idx]

    model = Neuralnetwork(config)
    check_grad(model, x_train_sample, y_train_sample)