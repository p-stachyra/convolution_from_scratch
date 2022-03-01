import numpy as np

class FullyConnectedLayer:
    def __init__(self, inputs, n_nodes):
        self.inputs = inputs
        self.n_inputs = inputs.shape[0]
        self.weights = 0.1 * np.random.randn(self.n_inputs, n_nodes)
        self.biases = np.zeros((1, n_nodes))
        self.output = 0

    def forward(self):
        # from previous layer
        self.output = np.dot(self.inputs, self.weights) + self.biases

