import torch.nn as nn

from src.utils.convolutional_utils import calculate_params_conv_t


class Autoencoder(nn.Module):
    def __init__(self, input_size, input_channels=1):
        super(Autoencoder, self).__init__()

        kernel_size = 3

        # For the parameter search of the decoder the preferred kernel size is 3
        self.kernel_sizes = [3, 1, 2, 5]

        padding = 1
        stride = 2

        output_size = input_size
        layer_index = 0

        self.encoder = nn.Sequential()

        current_channels = input_channels
        next_channels = 16
        output_sizes = {-1: input_size}

        while input_size > 4:

            self.encoder.add_module(
                f"conv{layer_index}", nn.Conv2d(current_channels, next_channels, 3, stride=2, padding=1)
            )
            self.encoder.add_module(f"batch_norm{layer_index}", nn.BatchNorm2d(next_channels))
            self.encoder.add_module(f"relu{layer_index}", nn.ReLU(inplace=True))

            current_channels = next_channels
            next_channels = next_channels * 2

            input_size = input_size // 2
            output_size = round((output_size + 2 * padding - kernel_size) / stride + 1)
            output_sizes[layer_index] = output_size
            layer_index += 1

        # Decoder
        self.decoder = nn.Sequential()

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

            self.decoder.add_module(
                f"conv_trans{i}",
                nn.ConvTranspose2d(
                    current_channels,
                    next_channels,
                    kernel_size,
                    stride=stride,
                    padding=padding,
                    output_padding=padding_out,
                ),
            )
            self.decoder.add_module(f"batch_norm{i}", nn.BatchNorm2d(next_channels))

            if i == 0:
                self.decoder.add_module(f"sigmoid{i}", nn.Sigmoid())
            else:
                self.decoder.add_module(f"relu{i}", nn.ReLU(inplace=True))
            current_channels = next_channels

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def build_model_pt(input_size=224):

    return Autoencoder(input_size=input_size)


if __name__ == "__main__":
    build_model_pt()
