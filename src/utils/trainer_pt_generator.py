import os
from typing import Any, Dict, List, Union

import torch
import torchvision.transforms.functional as TF
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

from src.utils.model_types import ModelType
from src.utils.trainer_pt import ModelTrainerPT


class ModelTrainerGeneratorPT(ModelTrainerPT):
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
        super(ModelTrainerGeneratorPT, self).__init__(
            model, criterion, optimizer, metrics, model_type, experiment_name, run_name
        )

    def calculate_loss(self, features, labels, outputs):
        """Wrapper to calculate various loss types"""
        return self.criterion(outputs, features)

    def log_results(self, epoch, loss, features, labels, outputs):
        print(f"At epoch {epoch}: loss={loss}")

        dst = f"results/epoch_{epoch}"
        os.makedirs(dst, exist_ok=True)
        if self.model_type == ModelType.GENERATOR:
            for i, output in enumerate(outputs):
                output = output.clamp(0, 1)
                img = TF.to_pil_image(output)
                img.save(os.path.join(dst, f"result_image_{i}.png"))

    def evaluate(self):
        return
