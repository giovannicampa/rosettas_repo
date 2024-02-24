from typing import Any, Dict, List, Union

import torch
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

from src.utils.trainer_base import ModelTrainerBase


class ModelTrainerPT(ModelTrainerBase):
    """Wraps a PyTorch model for training with convenient methods."""

    def __init__(
        self,
        model: torch.nn.Module,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        metrics: List[Any] = None,
    ) -> None:
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.metrics = metrics

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
                loss = self.criterion(outputs, labels)
                print(loss)

                # Backward and optimize
                loss.backward()
                self.optimizer.step()

    def evaluate(
        self, test_loader: torch.utils.data.DataLoader, device: str = "cpu"
    ) -> Dict[str, Union[float, torch.Tensor]]:
        """Evaluates the model on provided test data loader."""
        self.model.eval()

        total_correct = 0
        total_samples = 0
        all_labels = []
        all_preds = []

        with torch.no_grad():
            for features, labels in test_loader:
                features, labels = features.to(device), labels.to(device)
                outputs = self.model(features)

                _, predicted = torch.max(outputs.data, 1)
                total_samples += labels.size(0)
                total_correct += (predicted == labels).sum().item()

                # Store predictions and labels for further metrics calculation
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())

                # Calculate and accumulate evaluation metrics

        # Calculate accuracy
        accuracy = total_correct / total_samples

        # Calculate precision, recall, and F1 score using sklearn
        precision = precision_score(all_labels, all_preds, average="weighted")
        recall = recall_score(all_labels, all_preds, average="weighted")
        f1 = f1_score(all_labels, all_preds, average="weighted")

        # Calculate confusion matrix
        conf_matrix = confusion_matrix(all_labels, all_preds)

        # Store metrics in the dictionary
        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "confusion_matrix": conf_matrix,
        }

        return metrics

    def predict(
        self, data_loader: torch.utils.data.DataLoader, device: str = "cpu"
    ) -> List[Any]:
        """Predicts on new data using the trained model."""
        self.model.eval()
        predictions: List[Any] = []

        with torch.no_grad():
            for features in data_loader:
                features = features.to(device)
                predictions.extend(self.model(features).cpu().tolist())

        return predictions
