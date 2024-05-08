from abc import abstractmethod
from typing import Any, List

import torch

from src.utils.model_types import ModelType
from src.utils.trainer_base import ModelTrainerBase


class ModelTrainerPT(ModelTrainerBase):
    """Wraps a PyTorch model for training with convenient methods."""

    def __init__(
        self,
        model: torch.nn.Module = None,
        criterion: torch.nn.Module = None,
        optimizer: torch.optim.Optimizer = None,
        metrics: List[Any] = None,
        model_type: ModelType = ModelType.CLASSIFIER,
        experiment_name: str = "model_name",
        run_name: str = "pytorch",
    ) -> None:
        if run_name is None:
            run_name = "pytorch"
        super(ModelTrainerPT, self).__init__(experiment_name=experiment_name, run_name=run_name, model_type=model_type)
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.metrics = metrics
        self.model_type = model_type

    def train(
        self,
        train_loader: torch.utils.data.DataLoader,
        epochs: int,
        device: str = "cpu",
    ) -> None:
        """Trains the model on provided training data loader."""
        self.model.to(device)

        # Puts model in training mode (Enables Dropout Layers...)
        self.model.train()

        for epoch in range(epochs):
            for batch_idx, (features, labels) in enumerate(train_loader):
                features, labels = features.to(device), labels.to(device)

                # Forward pass
                self.optimizer.zero_grad()
                outputs = self.model(features)
                loss = self.calculate_loss(features, labels, outputs)
                if batch_idx == 0:
                    self.log_results(epoch, loss, features, labels, outputs)

                # Backward and optimize
                loss.backward()
                self.optimizer.step()

    @abstractmethod
    def calculate_loss(self, features, labels, outputs):
        pass

    @abstractmethod
    def log_results(self, epoch, loss, features, labels, outputs):
        pass

    def predict(self, data_loader: torch.utils.data.DataLoader, device: str = "cpu") -> List[Any]:
        """Predicts on new data using the trained model."""
        self.model.eval()
        predictions: List[Any] = []

        with torch.no_grad():
            for features in data_loader:
                features = features.to(device)
                predictions.extend(self.model(features).cpu().tolist())

        return predictions
