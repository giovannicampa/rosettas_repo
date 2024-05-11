from typing import List

import numpy as np

common_strides = [1, 2, 3]
common_padding = [0, 1, 2]


def calculate_params_conv(input_size: int, output_size: int, kernel_sizes: List):
    """Given a desired input and output shape for the inverse"""

    for kernel_size in kernel_sizes:
        for stride in common_strides:
            if stride > kernel_size:
                continue
            for padding in common_padding:
                calculated_output = np.floor((input_size + 2 * padding - kernel_size) / stride) + 1
                if calculated_output == output_size:
                    return stride, kernel_size, padding

    return None, None, None


def calculate_params_conv_t(input_size: int, output_size: int, kernel_sizes: List):
    """Given a desired input and output shape for the inverse"""

    for kernel_size in kernel_sizes:
        for stride in common_strides:
            if stride > kernel_size:
                continue
            for padding in common_padding:
                for padding_out in common_padding:
                    if padding_out > stride:
                        continue
                    calculated_output = (input_size - 1) * stride - 2 * padding + kernel_size + padding_out
                    if calculated_output == output_size:
                        return stride, kernel_size, padding, padding_out

    return None, None, None, None
