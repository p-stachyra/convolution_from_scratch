# global imports
import numpy as np

# project imports
from Convolution import Convolution
from ActivationHidden import ActivationHidden
from Pooling import Pooling
from Normalization import Normalization

class Layer:

	def __init__(self, array, kernels):
		# input attributes
		self.array = array
		self.feature_maps = [fm for fm in np.dsplit(array, array.shape[-1])]
		self.kernels = kernels

		# create lists for the processed feature maps
		# The convolved arrays: attribute
		self.convolved = []
		# The arrays after activation: attribute
		self.activated = []
		# The shrank arrays - after pooling: attribute
		self.pooled = []
		# The convolved arrays attribute
		self.normalized = []

	def transform(self):

		# convolve each feature map (color channel in the image)
		for fmap, kernel in zip(self.feature_maps, self.kernels):
			convolved_feature_map = Convolution.convolve(fmap.reshape(fmap.shape[0], fmap.shape[1]), kernel)
			self.convolved.append(convolved_feature_map)
			activated = ActivationHidden(convolved_feature_map).relu()
			self.activated.append(activated)
			pooled = Pooling(activated, kernel).pooling()
			self.pooled.append(pooled)
			normalized = Normalization(pooled).normalize()
			self.normalized.append(normalized)

		z = len(self.normalized)
		x, y = self.normalized[0].shape

		return np.array(self.normalized).flatten().reshape(z * x * y)





		