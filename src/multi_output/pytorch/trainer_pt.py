from typing import Any, List

import torch

from src.utils.trainer_pt import ModelTrainerPT


class ModelTrainerPTMultiOutput(ModelTrainerPT):
    """Wraps a PyTorch model for training with convenient methods."""

    def __init__(
        self,
        model: torch.nn.Module,
        criteria: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        metrics: List[Any] = None,
        model_type: str = None,
        experiment_name: str = None,
        run_name: str = None,
    ) -> None:
        super(ModelTrainerPTMultiOutput, self).__init__(
            model=model, optimizer=optimizer, experiment_name=experiment_name, run_name=run_name, model_type=model_type
        )
        self.model = model
        self.criteria = criteria
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

            for batch_idx, (features, multi_labels) in enumerate(train_loader):
                features = features.to(device)
                multi_labels = [
                    label.to(device) for label in multi_labels
                ]  # Assuming multi_labels is a list of tensors

                # This squeezes the dimension of size 1
                multi_labels = torch.stack(multi_labels).squeeze(1)
                multi_labels = multi_labels.transpose(0, 1)

                # Forward pass
                self.optimizer.zero_grad()
                multi_outputs = self.model(features)

                # Assuming you have a list of loss functions corresponding to each task
                # For example, self.criterion = [loss_fn1, loss_fn2, ...]
                total_loss = 0
                for output, label, criterion in zip(multi_outputs.transpose(1, 0), multi_labels, self.criteria):
                    loss = criterion(output, label)
                    total_loss += loss

                print(total_loss)

                # Backward pass and optimize
                total_loss.backward()
                self.optimizer.step()

    def evaluate(self, test_loader: torch.utils.data.DataLoader, device: str = "cpu"):
        self.model.eval()  # Set the model to evaluation mode
        total_loss = 0.0
        total_samples = 0

        self.model.to(device)

        for batch_idx, (features, multi_labels) in enumerate(test_loader):
            features = features.to(device)
            multi_labels = [label.to(device) for label in multi_labels]  # Assuming multi_labels is a list of tensors

            # This squeezes the dimension of size 1
            multi_labels = torch.stack(multi_labels).squeeze(1)
            multi_labels = multi_labels.transpose(0, 1)

            # Forward pass
            self.optimizer.zero_grad()
            multi_outputs = self.model(features)
            total_samples += multi_outputs.shape[0]

            for output, label, criterion in zip(multi_outputs.transpose(1, 0), multi_labels, self.criteria):
                loss = criterion(output, label)
                total_loss += loss

        average_loss = total_loss / total_samples
        return average_loss

    def predict(self, data_loader: torch.utils.data.DataLoader, device: str = "cpu") -> List[Any]:
        """Predicts on new data using the trained model."""
        self.model.eval()
        predictions: List[Any] = []

        with torch.no_grad():
            for batch_idx, (features, multi_labels) in enumerate(data_loader):
                features = features.to(device)
                predictions.extend(self.model(features).cpu().tolist())

        return predictions

    def calculate_loss(self, features, labels, outputs):
        """Wrapper to calculate various loss types"""
        return self.criterion(outputs, labels)

    def log_results(self, epoch, loss, features, labels, outputs):
        print(f"At epoch {epoch}: loss={loss}")
