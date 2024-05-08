from typing import Any, Dict, List, Union

import torch
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

from src.utils.model_types import ModelType
from src.utils.trainer_pt import ModelTrainerPT


class ModelTrainerClassifierPT(ModelTrainerPT):
    """Wraps a PyTorch model for training with convenient methods."""

    def __init__(
        self,
        model: torch.nn.Module,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        metrics: List[Any] = None,
        model_type: ModelType = ModelType.CLASSIFIER,
        experiment_name: str = None,
        run_name: str = None,
    ) -> None:
        super(ModelTrainerClassifierPT, self).__init__(
            model, criterion, optimizer, metrics, model_type, experiment_name, run_name
        )

    def calculate_loss(self, features, labels, outputs):
        """Wrapper to calculate various loss types"""
        return self.criterion(outputs, labels)

    def log_results(self, epoch, loss, features, labels, outputs):
        print(f"At epoch {epoch}: loss={loss}")

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
