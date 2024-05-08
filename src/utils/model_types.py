from enum import Enum, auto


class ModelType(Enum):
    CLASSIFIER = auto()
    REGRESSOR = auto()
    GENERATOR = auto()
