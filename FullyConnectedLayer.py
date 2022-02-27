import cv2
import numpy as np

from Layer import Layer

class FullyConnectedLayer:
    def __init__(self, inputs, n_nodes):
        self.inputs = inputs
        self.n_inputs = len(inputs)
        self.weights = 0.1 * np.random.randn(self.n_inputs, n_nodes)
        self.biases = np.zeros((1, n_nodes))
        self.output = 0

    def forward(self):
        # from previous layer
        self.output = np.dot(self.inputs, self.weights) + self.biases

