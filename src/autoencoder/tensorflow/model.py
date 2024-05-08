from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Input
from tensorflow.keras.models import Model, Sequential


class Autoencoder(Model):
    def __init__(self):

        super(Autoencoder, self).__init__()

        self.encoder = Sequential(
            [
                Input((28, 28, 1)),
                Conv2D(filters=16, kernel_size=3, strides=2, padding="same", activation="relu"),
                Conv2D(filters=32, kernel_size=3, strides=2, padding="same", activation="relu"),
                Conv2D(filters=64, kernel_size=7, padding="valid", activation="relu"),
            ]
        )

        self.decoder = Sequential(
            [
                Conv2DTranspose(filters=32, kernel_size=7, padding="valid", activation="relu"),
                Conv2DTranspose(filters=16, kernel_size=3, strides=2, padding="same", activation="relu"),
                Conv2DTranspose(filters=1, kernel_size=3, strides=2, padding="same", activation="sigmoid"),
            ]
        )

    def call(self, x):

        x = self.encoder(x)
        x = self.decoder(x)

        return x


def build_model_tf():

    return Autoencoder()


if __name__ == "__main__":
    model = build_model_tf()
