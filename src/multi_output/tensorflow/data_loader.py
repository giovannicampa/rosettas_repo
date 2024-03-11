import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.utils import Sequence, to_categorical

"""
Code from: https://towardsdatascience.com/building-a-multi-output-convolutional-neural-network-with-keras-ed24c7bc1178
"""


class MultiOutputDataGenerator(Sequence):
    def __init__(
        self,
        dataframe,
        outputs,
        batch_size=32,
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
        self.outputs = outputs
        self.batch_size = batch_size
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

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.floor(len(self.dataframe) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data"""
        # Generate indexes of the batch
        batch_indexes = self.indexes[
            index * self.batch_size : (index + 1) * self.batch_size
        ]

        # Initialize X (images) and y (labels)
        X = np.empty((self.batch_size, *self.img_size, 3))
        y = {
            feature: np.empty((self.batch_size), dtype=float)
            for feature in self.outputs
        }

        # Generate data
        for i, idx in enumerate(batch_indexes):

            img_path = self.dataframe.iloc[idx][self.image_column_name]
            img = load_img(img_path, target_size=self.img_size)
            X[i,] = (
                img_to_array(img) / 255.0
            )

            for feature in self.outputs:
                y[feature][i] = self.dataframe.iloc[idx][feature]

                # For categorical features
                # y[feature][i] = self.label_encoders[feature].transform(
                #     [self.dataframe.iloc[idx][feature]]
                # )[0]

        # Convert labels to categorical
        # y = {
        #     feature: to_categorical(
        #         y[feature], num_classes=len(self.label_encoders[feature].classes_)
        #     )
        #     for feature in self.outputs
        # }

        return X, y

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
