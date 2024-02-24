from tensorflow.keras.layers import Dense, Flatten, Input
from tensorflow.keras.models import Model


def build_model_tf(input_size: tuple, num_classes):

    inputs = Input(input_size)
    x = Flatten()(inputs)
    x = Dense(activation="relu", units=256)(x)
    x = Dense(activation="relu", units=128)(x)
    x = Dense(activation="softmax", units=num_classes)(x)

    model = Model(inputs=inputs, outputs=x)

    return model


if __name__ == "__main__":

    model = build_model_tf()
