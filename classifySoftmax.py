import cv2
import numpy as np
import sys

# local imports
from ActivationOutput import ActivationOutput
from Network import Network
from Transformation import Transformation
from Loss import Loss

# import kernel
from kernels import kernel_vertical

# helper function for preparing the kernel
def replicateKernel(kernel, image):
    return [kernel for i in range(image.shape[-1])]

def main():

    # read the input from the command line
    filename = sys.argv[1]
    image = cv2.imread(filename)

    # the kernel taken from a separate source file
    kernels = replicateKernel(kernel_vertical, image)

    # traverse thorough the initial operations on the input:
    # - convolution
    # - activation
    # - pooling
    # - normalization
    layer = Transformation(image, kernels)

    # the output of all of the initial operations
    transformed = layer.transform()

    # save the outputs of each operations in a visualized form
    cv2.imwrite("preview_images/Image_convolved.PNG", np.dstack(layer.convolved))
    cv2.imwrite("preview_images/Image_activated.PNG", np.dstack(layer.activated))
    cv2.imwrite("preview_images/Image_pooled.PNG", np.dstack(layer.pooled))
    cv2.imwrite("preview_images/Image_normalized.PNG", np.dstack(layer.normalized))

    io_sequence = transformed
    for i in range(5):
        fcl = Network(io_sequence, 5, [0, 0, 1, 0, 0])
        fcl.forward()
        output_for_softmax = fcl.output
        io_sequence = fcl.output.flatten()

    # The probabilities for the classes - using Softmax
    probabilities = ActivationOutput(output_for_softmax).softmax()

    # Computing loss
    true_output = np.array([0, 0, 1, 0, 0])
    objective_function = Loss(probabilities, true_output).categoricalCrossEntropy()

    # Output to STDOUT
    print("Probabilities for each class:")
    for p in list(probabilities):
        for prob in p:
            print(round(prob, 3))

    print("Loss:", objective_function)

if __name__ == "__main__":
    main()