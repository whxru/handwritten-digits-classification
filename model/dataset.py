import numpy as np


class Dataset:
    """
    Class for dealing with the structure of given data file
    """

    def __init__(self, data: np.ndarray, label: np.ndarray):
        self._data = data.copy()
        self._label = data.copy()

    @property
    def data(self):
        return self._data.copy()

    @property
    def labels(self):
        return self._label.copy()

    def sub_dataset(self, digit: int):
        """
        Extract the data and labels of given digits
        :param digit: the digit whose raw_data and labels are expected to be obtained
        :return: (dataset of the given digit, dataset of other digits)
        """

        raise NotImplementedError

    @staticmethod
    def load_matrix(fp: str) -> np.ndarray:
        with open(fp, 'r') as data_f:
            res = []
            for s in data_f:
                row = []
                for val_str in s.split():
                    row.append(float(val_str))
                res.append(row)
            return np.array(res)


if __name__ == '__main__':
    print(Dataset.load_matrix('../data/digits4000_digits_labels.txt').shape)
    print(Dataset.load_matrix('../data/digits4000_digits_vec.txt').shape)
