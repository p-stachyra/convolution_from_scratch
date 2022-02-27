from numpy.lib.stride_tricks import sliding_window_view

class Convolution:

    def convolve(feature_map, kernel):
        # Also known as rolling or moving window, the window slides across all dimensions
        # of the array and extracts subsets of the array at all window positions.
        # safe function to operate on a multidimensional array for
        # sliding a window and performing computations on that window
        multiplied = sliding_window_view(feature_map, kernel.shape) * kernel

        # perform summing at the internal,
        # first level of the dimension
        # needs to be reshaped first

        # reshape(depth, windows_in_inputs, cells_in_kernels)
        # depth - the dimensionality tackled by the function
        # windows_in_inputs - the end number of windows (to how many spots the kernel will be overlaid)
        # cells_in_kernels - the number of individual values within a kernel

        to_be_summed = multiplied.reshape(
            (feature_map.shape[0] + 1 - kernel.shape[0]) * (feature_map.shape[1] + 1 - kernel.shape[1]),
            kernel.shape[0] * kernel.shape[1])

        # perform summing for each array of a single cell's values
        summed = to_be_summed.sum(axis=1)

        # to produce readable feature maps (which will be the convolved feature maps of the previous layer)
        # the array must be reshaped: the number of these will be the same as the number of feature maps
        # the number of rows will be the number of rows of the input - 1 (as the multiplication always shares a row)
        # and the number of columns will be the number of columns of the input
        # - 1 (as the multiplication always shares a column)
        output = summed.reshape((feature_map.shape[0] + 1 - kernel.shape[0]),
                                (feature_map.shape[1] + 1 - kernel.shape[1]))

        return output
