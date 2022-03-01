import numpy as np

class Loss:
    def __init__(self, estimations, true_values):
        self.estimations = estimations
        self.true_values = true_values

    def categoricalCrossEntropy(self):
        n_outputs = len(self.estimations)
        # to avoid error : [log(0)]
        # do not allow for values = 0, values = 1 will be converted to almost 1
        adjusted = np.clip(self.estimations, 1e-7, 1 - 1e-7)

        correct = adjusted[range(n_outputs), self.true_values]

        # negative log likelihoods
        sample_losses = -np.log(correct)
        loss = np.mean(sample_losses)
        return loss