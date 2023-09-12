from enum import Enum


class NormalizeMethods(Enum):
    ZeroOneScaler = 1
    Standardize = 2
    StandardizeImageNet = 3
    none = -1


class ProjectMethods(Enum):
    Maximum = 1
    Mean = 2
    Focus = 3
    Channel = 4
    none = -1


class PredictMode(Enum):
    Standard = 1
    GetPrediction = 2
    GetEmbedding = 3
