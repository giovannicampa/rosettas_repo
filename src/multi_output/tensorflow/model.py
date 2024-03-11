from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import (
    Activation,
    BatchNormalization,
    Dense,
    Dropout,
    GlobalAveragePooling2D,
    Input,
)
from tensorflow.keras.models import Model


class MultiOutputModel:
    """
    Used to generate our multi-output model. This CNN contains three branches, one for age, other for
    sex and another for race. Each branch contains a sequence of Convolutional Layers that is defined
    on the make_default_hidden_layers method.
    """

    def __init__(self, features):
        self.features = features

    def make_default_hidden_layers(self, inputs):
        """
        Used to generate a default set of hidden layers. The structure used in this network is defined as:

        Conv2D -> BatchNormalization -> Pooling -> Dropout
        """

        base_model = ResNet50(
            weights="imagenet", include_top=False, input_tensor=inputs
        )

        # Freeze the layers of the base model
        for layer in base_model.layers:
            layer.trainable = False

        x = base_model.output
        x = GlobalAveragePooling2D()(x)  # Convert the MxNxC feature map to a 1xC vector
        x = Dense(1024, activation="relu")(x)  # Add a fully-connected layer

        return x

    def build_feature_branch(self, x, feature):
        x = Dense(128)(x)
        x = Activation("relu")(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        x = Dense(1)(x)
        x = Activation("linear", name=feature)(x)
        return x

    def assemble_full_model(self, input_shape):
        """
        Used to assemble our multi-output model CNN.
        """

        input_shape = input_shape
        inputs = Input(shape=input_shape)
        embeddings = self.make_default_hidden_layers(inputs)

        outputs = []
        for feature in self.features:
            output_i = self.build_feature_branch(x=embeddings, feature=feature)
            outputs.append(output_i)

        model = Model(
            inputs=inputs,
            outputs=outputs,
            name="face_net",
        )
        return model


def build_model_tf(input_size: tuple, features):

    model = MultiOutputModel(features).assemble_full_model(input_size)

    return model
