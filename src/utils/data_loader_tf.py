from typing import List, Tuple

import numpy as np
import tensorflow as tf
from PIL import Image
from sklearn.preprocessing import OneHotEncoder

"""
Code source: https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
"""


class DataGenerator(tf.keras.utils.Sequence):
    "Generates data for Keras"

    def __init__(
        self,
        image_filenames: List[str],
        labels: List[int],
        batch_size: int = 32,
        dim: Tuple[int, int, int] = (32, 32, 32),
        n_channels: int = 1,
        n_classes: int = 10,
        shuffle: bool = True,
    ) -> None:
        """Initializes the data generator with the dataset parameters.

        Args:
            image_filenames (List[str]): List of paths to the images.
            labels (List[int]): List of labels corresponding to the images.
            batch_size (int, optional): The size of the batch. Defaults to 32.
            dim (Tuple[int, int, int], optional): The dimensions of the images. Defaults to (32, 32, 32).
            n_channels (int, optional): The number of channels in the images. Defaults to 1.
            n_classes (int, optional): The number of classes. Defaults to 10.
            shuffle (bool, optional): Whether to shuffle the indexes every epoch. Defaults to True.
        """

        self.image_filenames = image_filenames
        self.indexes = np.arange(len(self.image_filenames))
        self.dim = dim
        self.batch_size = batch_size

        self.encoder = OneHotEncoder(sparse_output=False)
        self.encoder.fit(np.array(labels).reshape(-1, 1))
        self.labels = labels  # Needs to be one hot encoded
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        "Denotes the number of batches per epoch"
        return int(np.floor(len(self.image_filenames) / self.batch_size))

    def __getitem__(self, index):
        # Generate indexes of the batch
        batch_indexes = self.indexes[
            index * self.batch_size : (index + 1) * self.batch_size
        ]

        # Slice the batch
        batch_image_filenames = [self.image_filenames[k] for k in batch_indexes]

        # Generate data
        X, y = self.__data_generation(batch_image_filenames)

        if self.labels is not None:  # If generator is used for training/evaluation
            return X, y
        else:  # If generator is used for prediction
            return X

    def on_epoch_end(self):
        "Updates indexes after each epoch"
        self.indexes = np.arange(len(self.image_filenames))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, image_filenames_temp):
        "Generates data containing batch_size samples"  # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size, self.n_classes), dtype=int)

        # Generate data
        for i, image_path in enumerate(image_filenames_temp):
            # Store sample
            img = Image.open(image_path)
            img = img.convert("L")
            img = np.array(img) / 255.0
            X[
                i,
            ] = np.resize(img, (*self.dim, self.n_channels))

            # Store class
            y[i] = self.encoder.transform(np.array(self.labels[i]).reshape(-1, 1))

        return X, y
