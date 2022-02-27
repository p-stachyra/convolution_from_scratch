import numpy as np

class ActivationOutput:
    def __init__(self, inputs):
        self.inputs = inputs

    def softmax(self):
        exp_values = np.exp(self.inputs - np.max(self.inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        return probabilities