import tensorflow as tf
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.multi_output.pytorch.data_loader import MultiOutputDataGeneratorPT
from src.multi_output.pytorch.model import build_model_pt
from src.multi_output.pytorch.trainer_pt import ModelTrainerPT
from src.multi_output.tensorflow.data_loader import MultiOutputDataGeneratorTF
from src.multi_output.tensorflow.model import build_model_tf
from src.utils.dataset_housing_to_df import load_dataset
from src.utils.trainer_args_parser import train_args_parser
from src.utils.trainer_tf import ModelTrainerTF

if __name__ == "__main__":

    args = train_args_parser()

    img_size = 224
    nr_epochs = args.nr_epochs
    batch_size = args.batch_size
    learning_rate = args.learning_rate

    dataframe = load_dataset()

    features = {
        "nr_bedrooms": {
            "type": int,
            "loss_tf": "mean_squared_error",
            "loss_pt": nn.MSELoss(),
        },
        "nr_bathrooms": {
            "type": int,
            "loss_tf": "mean_squared_error",
            "loss_pt": nn.MSELoss(),
        },
        "area": {
            "type": float,
            "loss_tf": "mean_squared_error",
            "loss_pt": nn.MSELoss(),
        },
        "price": {
            "type": float,
            "loss_tf": "mean_squared_error",
            "loss_pt": nn.MSELoss(),
        },
    }

    if "tensorflow" in args.model_types:
        # Train keras model
        model_tf = build_model_tf(
            input_shape=(img_size, img_size, 3), features=features
        )
        optimizer_tf = tf.keras.optimizers.Adam(learning_rate)
        trainer_tf = ModelTrainerTF(
            model=model_tf,
            loss={feature: val["loss_tf"] for feature, val in features.items()},
            optimizer=optimizer_tf,
        )

        batch_size = 32
        generator_tf = MultiOutputDataGeneratorTF(
            dataframe,
            features.keys(),
            batch_size,
            img_size,
            image_column_name="image_path_frontal",
        )

        trainer_tf.train(
            train_data=generator_tf,
            epochs=nr_epochs,
            batch_size=batch_size,
        )

    if "pytorch" in args.model_types:
        # Train pytorch model
        model_pt = build_model_pt(
            input_shape=(img_size, img_size, 3), features=features
        )
        optimizer_pt = torch.optim.Adam(model_pt.parameters(), lr=learning_rate)
        trainer_pt = ModelTrainerPT(
            model=model_pt,
            criteria={val["loss_pt"] for _, val in features.items()},
            optimizer=optimizer_pt,
        )
        generator = MultiOutputDataGeneratorPT(
            dataframe,
            features.keys(),
            img_size,
            image_column_name="image_path_frontal",
        )
        dataloader = DataLoader(generator, batch_size=2, shuffle=True)

        trainer_pt.train(dataloader, epochs=nr_epochs)
