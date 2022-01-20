import numpy as np

np.random.seed(999)


def sigmoid(x):
    """use sigmoid function as activation function
    """
    return 1.0 / (1.0 + np.exp(-x))


class Network:
    def __init__(self, num_layers, num_neurons_per_layer):
        """init the network
        Args:
            num_layers (int): the number of network layers, excluding the input layer
            num_neurons_per_layer (list): the number of per layer's neurons, including the input layer's neurons
        """
        assert num_layers == len(num_neurons_per_layer) - 1, 'don\'t match between num_layers and num_neurons_per_layer.'

        self.num_layers = num_layers
        self.num_neurons_per_layer = num_neurons_per_layer

        self.weights = []
        self.bias = []
        for i in range(self.num_layers):
            # init weight and bias of per weighted layer
            self.weights.append(np.random.normal(0.0, pow(self.num_neurons_per_layer[i + 1], -0.5), (self.num_neurons_per_layer[i + 1], self.num_neurons_per_layer[i])))
            self.bias.append(np.zeros((self.num_neurons_per_layer[i + 1], 1)))

    def forward(self, input):
        """forward propagation
        Args:
            input (np.ndarray): required input dimension is 2 and shape is [n, 1]
        Returns:
            output of the whole network
        """
        self.outputs = []  # store the output of per layer, including the input layer
        output = input
        self.outputs.append(output)
        for i in range(self.num_layers):
            if i == self.num_layers - 1:  # we support that the last layer will not be activated
                output = np.matmul(self.weights[i], output) + self.bias[i]
            else:
                output = sigmoid(np.matmul(self.weights[i], output) + self.bias[i])

            self.outputs.append(output)

        return self.outputs[-1]

    def backward(self, target, lr):
        """back propagation
        Args:
            target (np.ndarray): true label of a input sample, required shape is [n, 1]
            lr (float): learning rate
        """
        # error back propagation
        self.errors = []  # store errors of per layer, from latter to former, last layer's error is (y_hat - y)
        for i in range(self.num_layers):
            if i == 0:  # error of last layer
                self.errors.append(self.outputs[-1] - target)
            else:  # propagate error to former layer
                self.errors.append(np.matmul((self.weights[self.num_layers - i]).T, self.errors[i - 1]))

        # gradient descent
        for i in range(self.num_layers):
            if i == 0:
                self.weights[self.num_layers - 1 - i] -= lr * np.matmul(self.errors[i], (self.outputs[-1 - i - 1]).T)
                self.bias[self.num_layers - 1 - i] -= lr * self.errors[i]
            else:
                self.weights[self.num_layers - 1 - i] -= lr * np.matmul(self.errors[i] * self.outputs[-1 - i] * (1.0 - self.outputs[-1 - i]), (self.outputs[-1 - i - 1]).T)
                self.bias[self.num_layers - 1 - i] -= lr * (self.errors[i] * self.outputs[-1 - i] * (1.0 - self.outputs[-1 - i]))
