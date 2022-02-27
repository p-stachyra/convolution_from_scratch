import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import cv2

def main():

    kernel_horizontal = np.array([[-1, -1, -1],
                               [0, 0, 0],
                               [1, 1, 1]])

    kernel_vertical = np.array([[-1, 0, 1],
                               [-1, 0, 1],
                               [-1, 0, 1]])


    # prepare the image and kernels
    test_img = cv2.imread("f_15.JPG")
    kernels_horizontal = [kernel_horizontal for i in range(test_img.shape[-1])]
    kernels_vertical = [kernel_vertical for i in range(test_img.shape[-1])]

    # obtain separate feature maps
    feature_maps1 = prepareFeatureMaps(test_img)

    # convolve each layer
    convolved_horizontal = convolveMatrices(feature_maps1, kernels_horizontal)
    # convolve each layer
    convolved_vertical = convolveMatrices(feature_maps1, kernels_vertical)

    # turn the output into an image to evaluate
    stacked_horizontal = np.dstack(convolved_horizontal)
    # turn the output into an image to evaluate
    stacked_vertical = np.dstack(convolved_vertical)

    # save the images
    cv2.imwrite("horizontal_kernel_conv2.PNG", stacked_horizontal)
    cv2.imwrite("vertical_kernel_conv2.PNG", stacked_vertical)


def convolveMatrices(feature_maps, kernels):

    # The arrays resulting from the convolution will be stored here
    outputs = []
    # convolve each feature map (color channel in the image)
    for fmap, kernel in zip(feature_maps, kernels):
        outputs.append(convolve(fmap.reshape(fmap.shape[0], fmap.shape[1]), kernel))

    return outputs

def prepareFeatureMaps(image):
    # extract the feature maps: in an image this will be usually 3 layers for each color channel
    feature_maps = [fm for fm in np.dsplit(image, image.shape[-1])]
    return feature_maps

def convolve(feature_map, kernel):

    # Also known as rolling or moving window, the window slides across all dimensions
    # of the array and extracts subsets of the array at all window positions.
    # safe function to operate on a multidimensional array for
    # sliding a window and performing computations on that window
    multiplied = sliding_window_view(feature_map, kernel.shape) * kernel

    # perform summing at the internal, first level of the dimension
    # needs to be reshaped first

    # reshape(depth, windows_in_inputs, cells_in_kernels)
    # depth - the dimensionality tackled by the function
    # windows_in_inputs - the end number of windows (to how many spots the kernel will be overlaid)
    # cells_in_kernels - the number of individual values within a kernel


    to_be_summed = multiplied.reshape((feature_map.shape[0] + 1 - kernel.shape[0]) * (feature_map.shape[1] + 1 - kernel.shape[1]), kernel.shape[0] * kernel.shape[1])

    # perform summing for each array of a single cell's values
    summed = to_be_summed.sum(axis=1)

    # to produce readable feature maps (which will be the convolved feature maps of the previous layer)
    # the array must be reshaped: the number of these will be the same as the number of feature maps
    # the number of rows will be the number of rows of the input - 1 (as the multiplication always shares a row)
    # and the number of columns will be the number of columns of the input
    # - 1 (as the multiplication always shares a column)
    output = summed.reshape((feature_map.shape[0] + 1 - kernel.shape[0]), (feature_map.shape[1] + 1 - kernel.shape[1]))

    return output


if __name__ == "__main__":
    main()

