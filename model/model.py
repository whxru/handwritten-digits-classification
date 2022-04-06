import os
import numpy as np
from dataset import Dataset


class ClassificationModel:

    def __init__(self, training_dataset: Dataset = None, saved_name=''):
        assert saved_name is not None or training_dataset is not None

        self._training_dataset = training_dataset
        self._features = np.array([])
        self._param = np.array([])

        if saved_name != '':
            self.restore(saved_name)

    def save(self, name):
        dir_name = f'../saved_model/{name}'
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)
        np.save(f'{dir_name}/{name}#data.npy', self._training_dataset.data)
        np.save(f'{dir_name}/{name}#labels.npy', self._training_dataset.labels)
        np.save(f'{dir_name}/{name}#features.npy', self._features)
        np.save(f'{dir_name}/{name}#param.npy', self._param)

    def restore(self, name):
        dir_name = f'../saved_model/{name}'
        data = np.load(f'{dir_name}/{name}#data.npy')
        labels = np.load(f'{dir_name}/{name}#labels.npy')
        self._training_dataset = Dataset(data, labels)
        self._features = np.load(f'{dir_name}/{name}#features.npy')
        self._param = np.load(f'{dir_name}/{name}#param.npy')

    def extract_feature(self, save_to_feature: bool = False, save_to_dataset = False, method: str = None, output_shape: np.shape = None, options: dict = {}) -> np.ndarray:
        """
        Extract features from the raw data
        :param save_to_feature: whether save the extracted features to self._features or not
        :param save_to_dataset: whether save the extracted features to self._training_dataset or not
        :param method: optional, name for the feature extraction method
        :param output_shape: optional, shape for the expected feature
        :param options: optional, key-value pairs for the algorithm setting
        :return: extracted features
        """

        x_mat = self._training_dataset.data
        x_dim = self._training_dataset.data_dim
        x_num = self._training_dataset.label_num
        feature_mat = []

        if method.lower() == 'pca':
            # Parse algorithm parameter
            dim_output = 10 if 'dim_output' not in options else options['dim_output']
            output_shape = (x_num, dim_output)
            # Convert each row vec in data to a feature vec
            for i, x in enumerate(x_mat):
                print(i)
                mu = np.sum(x) / x_dim
                x = x.reshape((x.shape[0], 1))
                cov = np.dot(x - mu, (x - mu).T)
                eig_val, eig_vec = np.linalg.eig(cov)
                eig_items = [{
                    'val': eig_val[idx], 'vec': eig_vec[idx]
                } for idx in range(len(eig_val))]
                eig_items.sort(key=lambda item: item['val'], reverse=True)
                phi = np.array([eig_items[idx]['vec'] for idx in range(dim_output)]).T
                x_converted = np.dot(phi.T, x - mu).reshape(dim_output)
                feature_mat.append(x_converted)
            feature_mat = np.array(feature_mat).reshape(output_shape)

        if save_to_feature:
            self._features = feature_mat
        if save_to_dataset:
            self._training_dataset = Dataset(feature_mat, self._training_dataset.labels)
        return feature_mat

    def learn(self, alg: str = None, options: dict = {}):
        """
        Learn from the extracted features and do training
        :param alg: optional, name for the used classification algorithm
        :param options: optional, key-value pairs for the algorithm setting
        :return:
        """

        pass

    def predict(self, x: np.ndarray, alg: str = None, options: dict = {}) -> int:
        """
        Predict
        :param x: raw_data of the digit's picture
        :param alg: optional, name for the used classification algorithm
        :param options: optional, key-value pairs for the algorithm setting
        :return: The predicted digit
        """

        pass


if __name__ == '__main__':
    dataset = Dataset(Dataset.load_matrix('../data/digits4000_digits_vec.txt'), Dataset.load_matrix('../data/digits4000_digits_labels.txt'))
    dataset_for_digit_3 = dataset.sub_dataset(3)
    classifier = ClassificationModel(dataset_for_digit_3)
    classifier.save('digit-3')
    classifier = ClassificationModel(saved_name='digit-3')
    print(classifier.extract_feature(method='PCA', options={'dim_output': 15}))
