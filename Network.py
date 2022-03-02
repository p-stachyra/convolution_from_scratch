import numpy as np

from ActivationHidden import ActivationHidden

class Network:
    def __init__(self, inputs, n_nodes, y):

        # X and y
        self.inputs = inputs
        self.n_inputs = inputs.shape[0]
        self.y = y

        # weights and biases attributes
        self.weights = 0.1 * np.random.randn(self.n_inputs, n_nodes)
        self.biases = np.zeros((1, n_nodes))

        # output attribute
        self.output = 0

    def forward(self):
        # from previous layer
        self.output = np.dot(self.inputs, self.weights) + self.biases

    def backward(self):
        # backpropagate
        self.error = self.y - self.output
        self.delta = self.error * ActivationHidden(self.output).relu()

        # how much the hidden layer weights contribute to the output error
        self.weights_error = self.delta.dot(self.weights.T)
        # the derivative of ReLU to weights error
        self.weights_delta = self.weights_error * ActivationHidden(self.inputs).relu()

        # adjusting the weights
        self.weights += self.inputs.dot(self.weights_delta.flatten())

    def fit(self):
        self.forward()
        self.backward()

