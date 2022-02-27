import numpy as np

class Normalization:

    def __init__(self, feature_map):
        self.feature_map = feature_map

    def normalize(self):
        return (self.feature_map - np.mean(self.feature_map)) / np.std(self.feature_map)