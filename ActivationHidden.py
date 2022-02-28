import numpy as np

class ActivationHidden:

    def __init__(self, feature_map):
        self.inputs = feature_map

    def relu(self):
        output = np.maximum(0, self.inputs)
        return output