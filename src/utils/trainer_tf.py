from typing import Dict, List

import numpy as np
import tensorflow as tf

from src.utils.trainer_base import ModelTrainerBase


class ModelTrainerTF(ModelTrainerBase):
    """
    Wraps a TensorFlow model for training with convenient methods.

    Args:
      model: A TensorFlow model to train. (tf.keras.Model)
      loss: The loss function to use for training. (tf.keras.losses.Loss)
      optimizer: The optimizer to use for training. (tf.keras.optimizers.Optimizer)
      metrics: A list of metrics to evaluate during training. (List[tf.keras.metrics.Metric])
    """

    def __init__(
        self,
        model: tf.keras.Model,
        loss: tf.keras.losses.Loss,
        optimizer: tf.keras.optimizers.Optimizer,
        metrics: List[tf.keras.metrics.Metric] = None,
    ) -> None:
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.metrics = metrics

    def compile(self) -> None:
        """
        Compiles the model for training with specified loss, optimizer, and metrics.
        """
        self.model.compile(
            loss=self.loss, optimizer=self.optimizer, metrics=self.metrics
        )

    def train(
        self,
        train_data: tf.data.Dataset,
        epochs: int,
        batch_size: int,
        validation_data: tf.data.Dataset | None = None,
    ) -> tf.keras.callbacks.History:
        """
        Trains the model on provided training data.

        Args:
          train_data: A TensorFlow Dataset for training data. (tf.data.Dataset)
          epochs: Number of training epochs. (int)
          batch_size: Size of training batches. (int)
          validation_data: A TensorFlow Dataset for validation data (optional). (Optional[tf.data.Dataset])

        Returns:
          A dictionary containing evaluation metrics. (tf.keras.callbacks.History)
        """
        self.compile()
        history = self.model.fit(
            train_data,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
        )
        return self.model

    def evaluate(self, test_data: tf.data.Dataset) -> Dict[str, float]:
        """
        Evaluates the model on provided test data.

        Args:
          test_data: A TensorFlow Dataset for test data. (tf.data.Dataset)

        Returns:
          A dictionary containing evaluation metrics. (Dict[str, float])
        """
        return self.model.evaluate(test_data)

    def predict(self, data: tf.data.Dataset) -> np.ndarray:
        """
        Predicts on new data using the trained model.

        Args:
          data: A TensorFlow Dataset for prediction data. (tf.data.Dataset)

        Returns:
          Predictions made by the model. (np.ndarray)
        """
        return self.model.predict(data)
