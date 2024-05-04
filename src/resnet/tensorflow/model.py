import tensorflow as tf
from tensorflow.keras.layers import (
    Add,
    BatchNormalization,
    Conv2D,
    Dense,
    GlobalAveragePooling2D,
    Lambda,
    MaxPooling2D,
    ReLU,
)
from tensorflow.keras.models import Model, Sequential


class ResidualBlock(Model):
    def __init__(self, out_channels: int, downsample: bool):
        super(ResidualBlock, self).__init__()

        if downsample:
            self.conv1 = Conv2D(out_channels, kernel_size=3, strides=2, padding="same")
            self.shortcut = Sequential(
                [Conv2D(out_channels, kernel_size=1, strides=2, padding="same"), BatchNormalization()]
            )
        else:
            self.conv1 = Conv2D(out_channels, kernel_size=3, strides=1, padding="same")
            self.shortcut = Lambda(lambda x: x)

        self.conv2 = Conv2D(out_channels, kernel_size=3, strides=1, padding="same")
        self.bn1 = BatchNormalization()
        self.bn2 = BatchNormalization()

    def call(self, x):

        shortcut = self.shortcut(x)
        out = ReLU()(self.bn1(self.conv1(x)))
        out = ReLU()(self.bn2(self.conv2(out)))
        out = Add()([out, shortcut])
        return ReLU()(out)


class ResNet(tf.keras.Model):
    def __init__(
        self,
        in_channels,
        input_size,
        repeat,
        residual_block: tf.keras.Model,
        use_bottleneck: bool = None,
        num_classes: int = 10,
    ):
        super(ResNet, self).__init__()

        self.in_channels = in_channels

        if use_bottleneck:
            filters = [64, 256, 512, 1024, 2048]
        else:
            filters = [64, 64, 128, 256, 512]

        self.layer0 = Sequential(
            [
                Conv2D(
                    64, input_shape=input_size, kernel_size=7, strides=2, padding="same"
                ),  # , input_shape=input_shape
                BatchNormalization(),
                ReLU(),
                MaxPooling2D(pool_size=3, strides=2, padding="same"),
            ]
        )

        self.layer1 = Sequential()
        self.layer1.add(residual_block(filters[1], downsample=False))
        for i in range(1, repeat[0]):
            self.layer1.add(residual_block(filters[1], downsample=False))

        self.layer2 = Sequential()
        self.layer2.add(residual_block(filters[2], downsample=True))
        for i in range(1, repeat[1]):
            self.layer2.add(residual_block(filters[2], downsample=False))

        self.layer3 = Sequential()
        self.layer3.add(residual_block(filters[3], downsample=True))
        for i in range(1, repeat[2]):
            self.layer3.add(residual_block(filters[3], downsample=False))

        self.layer4 = Sequential()
        self.layer4.add(residual_block(filters[4], downsample=True))
        for i in range(1, repeat[3]):
            self.layer4.add(residual_block(filters[4], downsample=False))

        self.gap = GlobalAveragePooling2D()
        self.fc = Dense(num_classes, activation="softmax")

    def call(self, x):
        out = self.layer0(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.gap(out)
        out = self.fc(out)
        return out


def build_model_tf(input_size: tuple = (28, 28, 1), num_classes=10):

    return ResNet(
        in_channels=1,
        input_size=input_size,
        residual_block=ResidualBlock,
        repeat=[2, 2, 2, 2],
        use_bottleneck=False,
        num_classes=num_classes,
    )


if __name__ == "__main__":

    model = build_model_tf(input_size=(28, 28, 1), num_classes=10)
