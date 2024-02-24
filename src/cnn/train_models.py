from glob import glob

import numpy as np
import tensorflow as tf
import torch
import torch.nn as nn

# from src.cnn.pytorch.model import build_model_pt
from src.cnn.tensorflow.model import build_model_tf

# from src.utils.data_loader_pt import ImageDataset
from src.utils.data_loader_tf import DataGenerator

# from src.utils.trainer_pt import ModelTrainerPT
from src.utils.trainer_tf import ModelTrainerTF

if __name__ == "__main__":

    img_size = 28
    nr_epochs = 1
    batch_size = 2
    learning_rate = 0.001

    images = glob("datasets/Houses-dataset/Houses Dataset/*.jpg", recursive=True)
    images.sort()

    # Generate the labels from the file names. Classes can be bathroom, bedroom, frontal and kitchen
    labels = [image.split("/")[-1].split("_")[1].split(".")[0] for image in images]
    num_classes = len(np.unique(labels))

    # Train keras model
    model_tf = build_model_tf(
        input_size=(img_size, img_size, 1), num_classes=num_classes
    )
    optimizer_tf = tf.keras.optimizers.Adam(learning_rate)
    trainer_tf = ModelTrainerTF(
        model=model_tf,
        loss=tf.keras.losses.categorical_crossentropy,
        optimizer=optimizer_tf,
    )
    generator_tf = DataGenerator(
        image_filenames=images,
        labels=labels,
        n_classes=num_classes,
        batch_size=batch_size,
        dim=(img_size, img_size),
    )
    trainer_tf.train(
        train_data=generator_tf,
        epochs=nr_epochs,
        batch_size=batch_size,
    )

    # # Train pytorch model
    # model_pt = build_model_pt(input_size=(img_size, img_size), num_classes=num_classes)
    # optimizer_pt = torch.optim.Adam(model_pt.parameters(), lr=learning_rate)
    # trainer_pt = ModelTrainerPT(
    #     model=model_pt, criterion=nn.CrossEntropyLoss(), optimizer=optimizer_pt
    # )
    # generator = ImageDataset(
    #     image_filenames=images,
    #     labels=labels,
    #     batch_size=batch_size,
    #     dim=(img_size, img_size),
    # )

    # trainer_pt.train(generator, epochs=nr_epochs)
