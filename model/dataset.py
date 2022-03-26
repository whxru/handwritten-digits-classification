import numpy as np


class Dataset:
    """
    Class for dealing with the structure of given data file
    """

    def __init__(self, data_fp: str, label_fp: str):
        self._data_fp = data_fp
        self._label_fp = label_fp

        self._data = Dataset.load_matrix(data_fp)
        self._label = Dataset.load_matrix(label_fp)

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
