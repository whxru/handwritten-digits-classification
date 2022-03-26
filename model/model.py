import numpy as np
from .dataset import Dataset


class ClassificationModel:

    def __init__(self, training_dataset: Dataset):
        self._training_dataset = training_dataset
        self._features = None
        self._param = None

    def extract_feature(self, save: bool = False, method: str = None, output_shape: np.shape = None) -> np.ndarray:
        """
        Extract features from the raw data
        :param save: whether save the extracted features to self._features or not
        :param method: optional, name for the feature extraction method
        :param output_shape: optional, shape for the expected feature
        :return: extracted features
        """

        raise NotImplementedError

    def learn(self, alg: str = None, options: dict = {}):
        """
        Learn from the extracted features and do training
        :param alg: optional, name for the used classification algorithm
        :param options: optional, key-value pairs for the algorithm setting
        :return:
        """

        raise NotImplementedError

    def predict(self, x: np.ndarray, alg: str = None, options: dict = {}) -> int:
        """
        Predict
        :param x: raw_data of the digit's picture
        :param alg: optional, name for the used classification algorithm
        :param options: optional, key-value pairs for the algorithm setting
        :return: The predicted digit
        """

        raise NotImplementedError
