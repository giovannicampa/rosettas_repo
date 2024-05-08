from glob import glob

import numpy as np
import tensorflow as tf
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.resnet.pytorch.model import build_model_pt
from src.resnet.tensorflow.model import build_model_tf
from src.utils.data_loader_pt import ImageDataset
from src.utils.data_loader_tf import DataGenerator
from src.utils.model_types import ModelType
from src.utils.trainer_args_parser import train_args_parser
from src.utils.trainer_pt_classifier import ModelTrainerClassifierPT
from src.utils.trainer_tf import ModelTrainerTF

if __name__ == "__main__":

    args = train_args_parser()

    img_size = 224
    nr_epochs = args.nr_epochs
    batch_size = args.batch_size
    learning_rate = args.learning_rate

    images = glob("datasets/Houses-dataset/Houses Dataset/*.jpg", recursive=True)
    images.sort()

    # Generate the labels from the file names. Classes can be bathroom, bedroom, frontal and kitchen
    labels = [image.split("/")[-1].split("_")[1].split(".")[0] for image in images]
    num_classes = len(np.unique(labels))

    if "tensorflow" in args.model_types:
        # Train keras model
        model_tf = build_model_tf(input_size=(img_size, img_size, 1), num_classes=num_classes)
        optimizer_tf = tf.keras.optimizers.Adam(learning_rate)
        trainer_tf = ModelTrainerTF(
            model=model_tf,
            loss=tf.keras.losses.categorical_crossentropy,
            optimizer=optimizer_tf,
            model_type=ModelType.CLASSIFIER,
            experiment_name="resnet",
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

    if "pytorch" in args.model_types:
        # Train pytorch model
        model_pt = build_model_pt(num_classes=num_classes)
        optimizer_pt = torch.optim.Adam(model_pt.parameters(), lr=learning_rate)
        trainer_pt = ModelTrainerClassifierPT(
            model=model_pt,
            criterion=nn.CrossEntropyLoss(),
            optimizer=optimizer_pt,
            model_type=ModelType.CLASSIFIER,
            experiment_name="resnet",
        )
        generator = ImageDataset(
            image_filenames=images,
            labels=labels,
            dim=(img_size, img_size),
        )
        dataloader = DataLoader(generator, batch_size=2, shuffle=True)

        trainer_pt.train(dataloader, epochs=nr_epochs)
