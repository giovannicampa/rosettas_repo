import numpy as np
import torch
from PIL import Image
from sklearn.preprocessing import OneHotEncoder
from torch.utils.data import Dataset
from torchvision import transforms


class MultiOutputDataGeneratorPT(Dataset):
    def __init__(
        self,
        dataframe,
        features,
        img_size=224,
        shuffle=True,
        image_column_name="image_path",
    ):
        """
        dataframe: Your pandas DataFrame.
        image_dir: The directory where images are stored.
        outputs: A list of column names in the dataframe that are the output features.
        batch_size: Number of samples per batch.
        img_size: Size to which the images will be resized.
        shuffle: Whether to shuffle the data between epochs.
        """
        self.dataframe = dataframe.copy()
        self.features = features
        # self.batch_size = batch_size
        self.img_size = (img_size, img_size)
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.dataframe))
        self.image_column_name = image_column_name
        # self.label_encoders = {
        #     feature: LabelEncoder().fit(self.dataframe[feature])
        #     for feature in self.outputs
        # }
        if shuffle:
            np.random.shuffle(self.indexes)

        self.image_filenames = self.dataframe[image_column_name]
        self.encoder = OneHotEncoder(sparse_output=False)
        # self.encoder.fit(np.array(labels).reshape(-1, 1))
        self.dim = (img_size, img_size)

        # Create the composite transform
        self.transform = transforms.Compose(
            [
                transforms.Resize(self.dim),  # Resize the image
                transforms.ToTensor(),  # Convert the image to a tensor and scale to [0, 1]
            ]
        )

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):

        img_path = self.image_filenames[idx]
        image = Image.open(img_path)
        image = image.convert("L")

        if self.transform:
            image = self.transform(image)

        label = self.dataframe.iloc[idx][self.features].to_numpy(dtype=np.float64)
        label = torch.tensor(label, dtype=torch.float64, requires_grad=True)
        return image, label
