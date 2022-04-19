from compileall import compile_file
import os
import numpy as np
from dataset import Dataset
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.svm import SVC,LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pickle


class ClassificationModel:

    def __init__(self, training_dataset: Dataset = None, saved_name=''):
        assert saved_name is not None or training_dataset is not None

        self._training_dataset = training_dataset
        self._features = np.array([])
        self._param = np.array([])
        self._acc=[]


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

        if method.lower() == 'scale':
            scale_ratio = 0.5 if 'scale_ratio' not in options else options['scale_ratio']
            resampling_alg = Image.NEAREST if 'resampling_alg' not in options else {
                'nearest': Image.NEAREST,
                'box': Image.BOX,
                'bilinear': Image.BILINEAR,
                'hamming': Image.HAMMING,
                'bicubic': Image.BICUBIC,
                'lanczos': Image.LANCZOS
            }[options['resampling_alg'].lower()]
            for i, x in enumerate(x_mat):
                original_img_size = int(np.sqrt(x_dim))
                new_img_size = int(original_img_size * scale_ratio)
                original_img_mat = x.reshape((original_img_size, original_img_size))
                im = Image.fromarray(np.uint8(original_img_mat))
                new_im = im.resize((new_img_size, new_img_size), resample=resampling_alg)
                new_img_array = np.asarray(new_im).reshape(new_img_size * new_img_size)
                feature_mat.append(new_img_array)

        feature_mat = np.array(feature_mat)
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
        if alg == 'SVM':
            #use corss_validation to select the best parameter
            train_x, test_x, train_y, test_y = train_test_split(self._training_dataset.data, self._training_dataset.labels.ravel(), test_size=(1-options['test_ratio']))
            dir_name = f'../saved_model/svm_model/'
            if not os.path.exists(dir_name):
                os.mkdir(dir_name)
            if options['one_vs_all']:
                #para_grid={'C':[0.1,0.5,1,5],'gamma':[10,5,1,0.1],'kernel':['linear','poly','rbf']}
                para_grid={'C':[0.1,0.5],'gamma':[10,5],'kernel':['poly']}
                grid = GridSearchCV(SVC(),para_grid,refit=True,verbose=2)
                grid.fit(train_x,train_y)
                print(grid.best_estimator_)
                grid_predictions=grid.predict(test_x)
                #print(confusion_matrix(test_y,grid_predictions))
                print(classification_report(test_y,grid_predictions))
                file_path=dir_name+str(options['num'])+'model.pickle'
                with open(file_path,'wb') as fp:
                    pickle.dump(grid,fp)
                np.save(dir_name+str(options['num'])+'classification_report.npy',classification_report(test_y,grid_predictions))
                
                return accuracy_score(test_y,grid_predictions)
            else:
                #Since ovo will cause 10*9/2=45 classifers, if we use cross-validation, it will cause a lot. 
                clf = SVC(decision_function_shape='ovo',C=options['C'],gamma=options['gamma'],kernel=options['kernel'])
                clf.fit(train_x,train_y)
                ovo_prediction=clf.predict(test_x)
                
                return accuracy_score(test_y,ovo_prediction)

                
        return

    def predict(self, x: np.ndarray, alg: str = None, options: dict = {}) -> int:
        """
        Predict
        :param x: raw_data of the digit's picture
        :param alg: optional, name for the used classification algorithm
        :param options: optional, key-value pairs for the algorithm setting
        :return: The predicted digit
        """

        pass

def experiment(options):
    if options['one_vs_all']:
        acc_all=[]
        for i in range(0,options['class_num']):
            options['num']=i
            dataset_for_digit = dataset.sub_dataset(i)
            classifier = ClassificationModel(dataset_for_digit)
            classifier.save('digit-'+str(i))
            acc=classifier.learn(alg='SVM',options=options)
            acc_all.append(acc)
        plt.plot(range(0,options['class_num']),acc_all,label='SVM accuracy')
        dir_name = f'../saved_model/svm_model/'
        plt.savefig(dir_name+'accuracy_svm.jpg')
        plt.show()
        plt.close()
    else:
        classifier=ClassificationModel(dataset)
        classifier.extract_feature(method='scale', save_to_dataset=True, options={'scale_ratio': .2})
        acc=classifier.learn(alg='SVM', options=options)
        print("One vs One SVM accuracy is:",acc)
       
if __name__ == '__main__':
    dataset = Dataset(Dataset.load_matrix('../data/digits4000_digits_vec.txt'), Dataset.load_matrix('../data/digits4000_digits_labels.txt'))
    options={'one_vs_all':False,
        'test_ratio':0.3,
        'class_num':10,
        'C':0.1,
        'gamma':1,
        'kernel':'poly',
        'dim_reduction':False,
        'save_model':True
        }
    experiment(options)
    # print(classifier.extract_feature(method='PCA', options={'dim_output': 15}))
    

