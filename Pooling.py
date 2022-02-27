from numpy.lib.stride_tricks import sliding_window_view

class Pooling:

    def __init__(self, feature_map, extent):
        self.feature_map = feature_map
        self.window = extent

    def pooling(self):

        windows = sliding_window_view(self.feature_map, self.window.shape)

        adjusted_windows = windows.reshape(
            (self.feature_map.shape[0] + 1 - self.window.shape[0]) * (self.feature_map.shape[1] + 1 - self.window.shape[1]),
            self.window.shape[0] * self.window.shape[1])

        pooled = adjusted_windows.max(axis=1)

        output = pooled.reshape((self.feature_map.shape[0] + 1 - self.window.shape[0]),
                                (self.feature_map.shape[1] + 1 - self.window.shape[1]))

        return output
