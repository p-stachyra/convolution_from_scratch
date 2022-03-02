# global imports
import numpy as np

# project imports
from Convolution import Convolution
from ActivationHidden import ActivationHidden
from Pooling import Pooling
from Normalization import Normalization

class Transformation:

	def __init__(self, image, kernel, n_iter):
		# input attributes
		self.array = image
		self.feature_maps = [fm for fm in np.dsplit(image, image.shape[-1])]
		self.kernel = kernel
		self.n_iter = n_iter

		# create lists for the processed feature maps
		# The convolved arrays: attribute
		self.convolved = []
		# # The arrays after activation: attribute
		# self.activated = []
		# # The shrank arrays - after pooling: attribute
		# self.pooled = []
		# # The convolved arrays attribute
		# self.normalized = []
		self.output = 0

	def transform(self):

		# initial step
		# convolve each feature map (color channel in the image)
		aggregated_layer = 0
		for fmap in self.feature_maps:
			convolved_feature_map = Convolution(fmap.reshape(fmap.shape[0], fmap.shape[1]),
												self.kernel).convolve()
			# aggregate the values in 1 layer
			aggregated_layer += convolved_feature_map
			self.convolved.append(convolved_feature_map)

		# start with the initially aggregated layer
		# to be updated for each iteration
		io_sequence = aggregated_layer
		for i in range(self.n_iter):
			convolved = Convolution(aggregated_layer.reshape(aggregated_layer.shape[0], aggregated_layer.shape[1]),
												self.kernel).convolve()
			activated = ActivationHidden(convolved).relu()
			pooled = Pooling(activated, self.kernel).pooling()
			normalized = Normalization(pooled).normalize()
			io_sequence = normalized

		self.output = io_sequence
		return io_sequence





		