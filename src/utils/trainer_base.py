from abc import ABC, abstractmethod

import mlflow

from src.utils.model_types import ModelType


class ModelTrainerBase(ABC):
    def __init__(self, experiment_name: str = None, run_name: str = None, model_type: ModelType = None):
        self.setup_mlflow(experiment_name=experiment_name, run_name=run_name)
        assert isinstance(model_type, ModelType)

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def evaluate(self):
        pass

    @abstractmethod
    def predict(self):
        pass

    def setup_mlflow(self, experiment_name: str, run_name: str):
        mlflow.set_experiment(experiment_name)
        mlflow.start_run()
        mlflow.set_tag("mlflow.runName", run_name)

    def end_mlflow(self):
        mlflow.end_run()
