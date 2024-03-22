import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_default_tensor_type(torch.DoubleTensor)


class MultiOutputFeedforwardNeuralNet(nn.Module):
    def __init__(self, input_shape, features):
        super(MultiOutputFeedforwardNeuralNet, self).__init__()

        self.features = features

        self.flat_input_size = input_shape[0] * input_shape[1]
        # Shared layers
        self.fc1 = nn.Linear(self.flat_input_size, 256)
        self.fc2 = nn.Linear(256, 128)

        # Using ModuleDict to dynamically create output layers
        self.output_layers = nn.ModuleDict()

        for feature_key, feature_vals in self.features.items():

            # For the case of classification
            if feature_vals["type"] == "categorical":
                self.output_layers[feature_key] = nn.Linear(
                    128, feature_vals["num_classes"]
                )

            # For the case of regression
            else:
                self.output_layers[feature_key] = nn.Linear(128, 1)

    def forward(self, x):
        # Shared forward pass
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        outputs = []

        # Separate forward passes for each output
        for i, layer in self.output_layers.items():
            outputs.append(layer(x))

        outputs = (
            torch.stack(outputs, dim=1)
            .reshape(-1, len(self.features))
            .requires_grad_(True)
        )

        return outputs


def build_model_pt(input_shape, features):
    return MultiOutputFeedforwardNeuralNet(input_shape=input_shape, features=features)


if __name__ == "__main__":

    model = build_model_pt((3, 32, 32), 10, 5)
