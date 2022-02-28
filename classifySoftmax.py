import cv2
import numpy as np

from ActivationOutput import ActivationOutput
from FullyConnectedLayer import FullyConnectedLayer
from Layer import Layer

def main():

    kernel_horizontal = np.array([[-1, -1, -1],
                                  [0, 0, 0],
                                  [1, 1, 1]])

    kernel_vertical = np.array([[-1, 0, 1],
                                [-1, 0, 1],
                                [-1, 0, 1]])

    test_img = cv2.imread("f_15.JPG")

    kernels_horizontal = [kernel_horizontal for i in range(test_img.shape[-1])]
    kernels_vertical = [kernel_vertical for i in range(test_img.shape[-1])]

    layer = Layer(test_img, kernels_vertical)
    transformed = layer.transform()
    print(transformed)

    fcl = FullyConnectedLayer(transformed, 3)
    fcl.forward()

    probabilities = ActivationOutput(fcl.output).softmax()
    #print(probabilities)

if __name__ == "__main__":
    main()