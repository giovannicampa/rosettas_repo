import numpy as np
import torch
from PIL import Image
from sklearn.preprocessing import OneHotEncoder
from torch.utils.data import Dataset
from torchvision import transforms


class ImageDataset(Dataset):
    def __init__(self, labels, image_filenames, dim):

        self.unique_labels = sorted(set(labels))  # Sort to ensure consistency
        self.label_to_index = {
            label: index for index, label in enumerate(self.unique_labels)
        }
        self.labels = [self.label_to_index[label] for label in labels]

        self.image_filenames = image_filenames
        self.encoder = OneHotEncoder(sparse_output=False)
        self.encoder.fit(np.array(labels).reshape(-1, 1))
        self.dim = dim

        # Create the composite transform
        self.transform = transforms.Compose(
            [
                transforms.Resize(self.dim),  # Resize the image
                transforms.ToTensor(),  # Convert the image to a tensor and scale to [0, 1]
            ]
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        img_path = self.image_filenames[idx]
        image = Image.open(img_path)
        image = image.convert("L")

        if self.transform:
            image = self.transform(image)

        label = self.labels[idx]
        label = torch.tensor(label, dtype=torch.long)
        return image, label
