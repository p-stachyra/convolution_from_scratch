import numpy as np
import cv2
import sys

from ActivationOutput import ActivationOutput
from kernels import kernel_horizontal
from Loss import Loss
from Network import Network
from Transformation import Transformation

def main():
    # read the input from the command line
    filename = sys.argv[1]
    image = cv2.imread(filename)

    feature_maps = [fm for fm in np.dsplit(image, image.shape[-1])]
    true_output = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0])


    transformed = Transformation(image, kernel_horizontal, 2)
    transformed.transform()
    output = transformed.output.flatten()
    cv2.imwrite("preview_images/Image_convolved.PNG", np.dstack(transformed.convolved))

    network = Network(output, 10, true_output)

    for i in range(6):  # trains the NN 1000 times
        network.fit()
        # Yielding the probabilities for each estimation
        output_layer = ActivationOutput(network.output).softmax()
        # Computing loss
        objective_function = Loss(output_layer, true_output).categoricalCrossEntropy()
        print("Loss:", objective_function)

    # Output the final probabilities to STDOUT
    print("Probabilities for each class:")
    for p in output_layer.flatten():
            print(round(p, 3))

if __name__ == "__main__":
    main()








