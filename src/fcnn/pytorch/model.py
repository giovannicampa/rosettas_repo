import torch
import torch.nn as nn
import torch.nn.functional as F


class FeedforwardNeuralNet(nn.Module):
    def __init__(self, input_size, num_classes):
        super(FeedforwardNeuralNet, self).__init__()
        # First fully connected layer

        self.flat_input_size = input_size[0] * input_size[1]
        self.fc1 = nn.Linear(
            self.flat_input_size, 256
        )  # First layer takes the flattened image
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):

        x = x.view(-1, 784)  # Flatten input image
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def build_model_pt(input_size, num_classes):
    return FeedforwardNeuralNet(input_size=input_size, num_classes=num_classes)


if __name__ == "__main__":
    build_model_pt()
