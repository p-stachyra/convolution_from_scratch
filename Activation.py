import numpy as np

class Activation:

    def __init__(self, feature_map):
        self.inputs = feature_map

    def relu(self):
        output = np.maximum(0, self.inputs)
        return output