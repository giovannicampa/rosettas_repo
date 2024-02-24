from abc import ABC, abstractmethod


class ModelTrainerBase(ABC):
    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def evaluate(self):
        pass

    @abstractmethod
    def predict(self):
        pass

    def setup_mlflow():
        print("")
