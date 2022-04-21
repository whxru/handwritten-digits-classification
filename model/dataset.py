import numpy as np


class Dataset:
    """
    Class for dealing with the structure of given data file
    """

    def __init__(self, data: np.ndarray, labels: np.ndarray):
        self._data = data.copy()
        self._labels = labels.copy()

    @property
    def data(self):
        return self._data.copy()

    @property
    def data_dim(self):
        return self._data.shape[1]

    @property
    def label_num(self):
        return self._labels.shape[0]

    @property
    def labels(self):
        return self._labels.copy()

    def copy(self):
        return Dataset(self._data.copy(), self._labels.copy())

    def sub_dataset(self, digit: int, expected_labels=(1, -1)):
        """
        Replace the label of target digit and other digits and return a new Dataset
        :param digit: the digit whose raw_data and labels are expected to be obtained
        :param expected_labels: the labels (given to expected digit, other digits), whose default value is (1, -1)
        :return: New dataset where the label of target digit and other digits are respectively replaced
        """
        res_data, res_labels = self._data.copy(), self._labels.copy()
        for idx in range(self._labels.shape[0]):
            is_target_digit = self._labels[idx][0] == digit
            res_labels[idx][0] = expected_labels[0] if is_target_digit else expected_labels[1]

        return Dataset(res_data, res_labels)

    @staticmethod
    def load_matrix(fp: str) -> np.ndarray:
        with open(fp, 'r') as data_f:
            res = []
            for s in data_f:
                row = []
                for val_str in s.split():
                    row.append(float(val_str))
                res.append(row)
            print(np.array(res).shape)
            return np.array(res)

if __name__ == '__main__':
    dataset = Dataset(Dataset.load_matrix('../data/digits4000_digits_vec.txt'), Dataset.load_matrix('../data/digits4000_digits_labels.txt'))
    dataset_for_digit_3 = dataset.sub_dataset(3)
    print(np.count_nonzero(dataset_for_digit_3.labels == 1))
    print(np.count_nonzero(dataset_for_digit_3.labels == -1))
