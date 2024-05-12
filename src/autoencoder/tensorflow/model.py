from tensorflow.keras.layers import (
    Activation,
    BatchNormalization,
    Conv2D,
    Conv2DTranspose,
    Input,
    ZeroPadding2D,
)
from tensorflow.keras.models import Model, Sequential

from src.utils.convolutional_utils import calculate_params_conv_t


class Autoencoder(Model):
    def __init__(self, input_size, input_channels=1):

        super(Autoencoder, self).__init__()

        kernel_size = 3

        # For the parameter search of the decoder the preferred kernel size is 3
        self.kernel_sizes = [3, 1, 2, 5]

        padding = "same"
        stride = 2

        output_size = input_size
        layer_index = 0

        self.encoder = Sequential()

        current_channels = input_channels
        next_channels = 16
        output_sizes = {-1: input_size}

        while input_size > 4:

            if layer_index == 0:
                self.encoder.add(
                    Conv2D(
                        next_channels,
                        kernel_size,
                        strides=stride,
                        padding=padding,
                        input_shape=(input_size, input_size, current_channels),
                    )
                )
            else:
                self.encoder.add(Conv2D(next_channels, kernel_size, strides=stride, padding=padding))

            self.encoder.add(BatchNormalization())
            self.encoder.add(Activation("relu"))

            current_channels = next_channels
            next_channels = next_channels * 2

            input_size = input_size // 2

            output_size = self.encoder.layers[-1].output_shape[1]
            output_sizes[layer_index] = output_size
            layer_index += 1

        self.decoder = Sequential()

        input_size = output_size

        for i in range(layer_index - 1, -1, -1):
            if i == 0:
                next_channels = input_channels
            else:
                next_channels = current_channels // 2

            output_size = output_sizes[i - 1]
            input_size = output_sizes[i]

            stride, kernel_size, padding, padding_out = calculate_params_conv_t(
                input_size=input_size, output_size=output_size, kernel_sizes=self.kernel_sizes
            )

            self.decoder.add(
                Conv2DTranspose(
                    filters=next_channels,  # Number of output channels
                    kernel_size=kernel_size,  # Size of the kernel
                    strides=stride,  # Stride of the convolution
                    output_padding=padding_out,  # Additional padding to ensure output size matches the target
                    padding="same",
                )
            )
            self.decoder.add(BatchNormalization())
            self.decoder.add(Activation("relu") if i == 0 else Activation("sigmoid"))
            current_channels = next_channels

    def call(self, x):

        x = self.encoder(x)
        x = self.decoder(x)

        return x


def build_model_tf(input_size=224):

    return Autoencoder(input_size=input_size)


if __name__ == "__main__":
    model = build_model_tf()
