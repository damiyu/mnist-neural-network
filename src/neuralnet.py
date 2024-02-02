import numpy as np
import util

class Activation():
    def __init__(self, activation_type = "sigmoid"):
        if activation_type not in ["sigmoid", "tanh", "ReLU", "output"]:
            raise NotImplementedError(f"{activation_type} is not implemented.")

        # Type of non-linear activation.
        self.activation_type = activation_type

        # Placeholder for input. This can be used for computing gradients.
        self.x = None

    def __call__(self, z): return self.forward(z)

    def forward(self, z):
        # Call the appropriate activation funtion g(a)
        if self.activation_type == "sigmoid": return self.sigmoid(z)
        elif self.activation_type == "tanh": return self.tanh(z)
        elif self.activation_type == "ReLU": return self.ReLU(z)
        elif self.activation_type == "output": return self.output(z)

    def backward(self, z):
        # Call the appropriate activation derivative function g'(a)
        if self.activation_type == "sigmoid": return self.grad_sigmoid(z)
        elif self.activation_type == "tanh": return self.grad_tanh(z)
        elif self.activation_type == "ReLU": return self.grad_ReLU(z)
        elif self.activation_type == "output": return self.grad_output(z)

    # Three different activation functions.
    def sigmoid(self, x): return 1 / (1 + np.exp(-1 * x))
    def tanh(self, x): return (2 / (1 + np.exp(-2 * x))) - 1
    def ReLU(self, x): return np.maximum(0, x)

    def output(self, x):
        # Exponentiate every a.
        x[x >= 100] = 100
        e_a = np.exp(x)

        # Collect the sums of each row of exp(a_j).
        e_sum = []
        for a in e_a: e_sum.append([np.sum(a)])
        return e_a / np.array(e_sum)

    # The derivates of the three activation functions
    def grad_sigmoid(self, x):
        sig = self.sigmoid(x)
        return sig * (1 - sig)
    def grad_tanh(self, x):
        tanh = self.tanh(x)
        return 1 - (tanh * tanh)
    def grad_ReLU(self, x):
        return 1 * (x > 0)

    def grad_output(self, x):
        return np.full(x.shape, 1)

class Layer():
    def __init__(self, in_units, out_units, activation):
        np.random.seed(42)
        self.w = 0.01 * np.random.random((in_units + 1, out_units))

        self.x = None    # Save the input to forward in this
        self.a = None    # Output without activation
        self.z = None    # Output After Activation
        self.activation = activation

        self.hasOld = False
        self.dp = None # Previous delta, used for momentum
        self.dw = 0  # Save the gradient w.r.t w in this. w already includes bias term

    def __call__(self, x): return self.forward(x)

    def forward(self, x, checkGradient, dim, constant):
        # Update the weights by the constant if we are checking the gradient.
        if checkGradient:
            if dim[1] == self.w.shape[0] and dim[2] == self.w.shape[1]:
                self.w[dim[3]][dim[4]] += constant

        # Save the inputs with a bias term x_0 = 1.
        bias = np.ones((x.shape[0], 1))
        self.x = np.concatenate((x, bias), axis = 1)

        # Get the output and actived output
        self.a = np.matmul(self.x, self.w)
        self.z = self.activation.forward(self.a)
        return self.z

    def backward(self, deltaCur, learning_rate, useMomentum, gamma, lamb1, lamb2, isHiddenLayer, gradReqd):
        # Calculate the new weights with matrix multiplication and addition.
        if isHiddenLayer: deltaCur = self.activation.backward(self.a) * np.transpose(deltaCur)[:,:-1]
        self.dp = (learning_rate * np.matmul(np.transpose(self.x), deltaCur))
        w_n = self.w + (learning_rate * np.matmul(np.transpose(self.x), deltaCur))

        # Regularization step to penalize overfitting with the derivates of C.
        w_n += lamb1 * np.sign(self.w)
        w_n += 2 * lamb2 * self.w

        # Include momentum in the update rule if applicable
        if useMomentum and self.hasOld: w_n += gamma * self.dp

        # Do not update the weights if we are only checking gradients.
        if not gradReqd: self.w = w_n
        self.hasOld = True

        # Calculate and save the previous deltas for backward pass use.
        self.dw = np.matmul(w_n, np.transpose(deltaCur))
        return (self.dw, np.matmul(np.transpose(self.x), deltaCur))

class Neuralnetwork():
    def __init__(self, config):
        self.layers = []  # Store all layers in this list.
        self.num_layers = len(config['layer_specs']) - 1  # Set num layers here
        self.x = None  # Save the input to forward in this
        self.y = None  # For saving the output vector of the model
        self.targets = None  # For saving the targets
        self.config = config

        # Add layers specified by layer_specs.
        for i in range(self.num_layers):
            if i < self.num_layers - 1:
                self.layers.append(Layer(config['layer_specs'][i], config['layer_specs'][i + 1], Activation(config['activation'])))
            elif i  == self.num_layers - 1:
                self.layers.append(Layer(config['layer_specs'][i], config['layer_specs'][i + 1], Activation("output")))

    def __call__(self, x, targets=None): return self.forward(x, targets)

    def loss(self, logits, targets):
        # Calculate cross-entropy loss by getting the negative sum of targets times the natural log of my probability distribution.
        return -np.sum(targets * np.log(logits))

    def forward(self, x, targets=None, modifyWeights=False, dim=(None, 0, 0, 0, 0), e_constant=0.0):
        # Initial forward pass of inputs to second layer.
        curLayer = self.layers[0]
        self.x = x
        self.y = curLayer.forward(x, modifyWeights, dim, e_constant)
        self.targets = targets

        # Compute forward pass on the hidden layers until the output layer.
        for i in range(1, self.num_layers):
            curLayer = self.layers[i]
            self.y = curLayer.forward(self.y, modifyWeights, dim, e_constant)

        # Calculate accuaracy and loss.
        t = np.argmax(targets, axis = 1) == np.argmax(self.y, axis = 1)
        return (t.sum(), self.loss(self.y, self.targets))

    def backward(self, gradReqd=False, dim=(None, 0, 0, 0, 0)):
        # Create the deltas of the output layer.
        curDelta, save_grad = self.targets - self.y, None

        # Get the lambdas for regularization if avaliable
        lamb1 = 0 if not "L1_penalty" in self.config else self.config["L1_penalty"]
        lamb2 = 0 if not "L2_penalty" in self.config else self.config["L2_penalty"]

        for i in range(self.num_layers - 1, -1, -1):
            # Get the layer and check if it is a hidden layer.
            curLayer = self.layers[i]
            hidden = i < self.num_layers - 1

            curDelta, grad = curLayer.backward(curDelta, self.config['learning_rate'], self.config['momentum'], self.config['momentum_gamma'], lamb1, lamb2, hidden, gradReqd)
            dX, dY = grad.shape
            if dim[0] == "output" and dX == dim[1] and dY == dim[2]: save_grad = grad
            elif dim[0] == "hidden" and dX == dim[1] and dY == dim[2]: save_grad = grad

        # Output the gradient at the specified weight dimensions when required.
        if gradReqd: return save_grad[dim[3]][dim[4]]