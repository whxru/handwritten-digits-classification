from compileall import compile_file
import os
import numpy as np
import time
from pkg_resources import cleanup_resources
from dataset import Dataset
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.svm import SVC,LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report,confusion_matrix, plot_confusion_matrix
from sklearn.decomposition import PCA
import pickle


options={'manually_one_vs_all':True,
    'official_svm':True,
    'OVO': True,
    'OVR': True,
    'test_ratio':0.3,
    'class_num':10,
    'C':0.1,
    'gamma':1,
    'kernel':'poly',
    'dim_reduction':True,
    'save_model':True,
    'extract_feature':True,
    'pca':True,
    'scale':True,
    'mixed':True
    }


class ClassificationModel:

    def __init__(self, training_dataset: Dataset = None, test_dataset: Dataset = None, saved_name=''):
        assert saved_name is not None or training_dataset is not None

        self._training_dataset = training_dataset
        self._test_dataset = test_dataset
        self._features = np.array([])
        self._features_test = np.array([])
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

    def show_im(self, img_idx=None, ax=None):
        if img_idx is None:
            img_idx = np.random.randint(low=0, high=self._training_dataset.label_num - 1)
        if ax is None:
            fig, ax = plt.subplots()
        img_mat = self._training_dataset.data[img_idx]
        img_length = int(np.sqrt(self._training_dataset.data_dim))
        ax.imshow(img_mat.reshape((img_length, img_length)))
        return img_idx

    def restore(self, name):
        dir_name = f'../saved_model/{name}'
        data = np.load(f'{dir_name}/{name}#data.npy')
        labels = np.load(f'{dir_name}/{name}#labels.npy')
        self._training_dataset = Dataset(data, labels)
        self._features = np.load(f'{dir_name}/{name}#features.npy')
        self._param = np.load(f'{dir_name}/{name}#param.npy')

    @staticmethod
    def stat_extract_feature(x_mat, method='pca', options={}, transformer=None):
        x_dim = x_mat.shape[1]
        feature_mat = []

        if method.lower() in ['scale', 'mixed']:
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

        if method.lower() in ['pca', 'mixed']:
            # Parse algorithm parameter
            dim_output = 10 if 'dim_output' not in options else options['dim_output']
            if method.lower() == 'mixed':
                x_mat = feature_mat.copy()
                x_dim = x_mat.shape[1]

            dim_output = min(dim_output, x_dim)
            if transformer is not None:
                feature_mat = transformer.transform(x_mat)
            else:
                pca = PCA(n_components=dim_output)
                transformer = pca
                feature_mat = pca.fit_transform(x_mat)

        return feature_mat, transformer

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
        t0 = time.time()
        transformer = None
        for data_type, x_mat in enumerate([self._training_dataset.data, self._test_dataset.data]):
            feature_mat, transformer = ClassificationModel.stat_extract_feature(x_mat, method=method, options=options, transformer=transformer)
            if save_to_feature:
                if data_type == 0:
                    self._features = feature_mat.copy()
                else:
                    self._features_test = feature_mat.copy()
            if save_to_dataset:
                if data_type == 0:
                    self._training_dataset = Dataset(feature_mat, self._training_dataset.labels)
                else:
                    self._test_dataset = Dataset(feature_mat, self._test_dataset.labels)
        print(f'Feature extraction time: {time.time() - t0} seconds')

    def learn(self, data_type: str, official_svm, class_num: int = 0, alg: str = None, options: dict = {}):
        """
        Learn from the extracted features and do training
        :param alg: optional, name for the used classification algorithm
        :param options: optional, key-value pairs for the algorithm setting
        :return:
        """
        if alg == 'SVM':
            #use corss_validation to select the best parameter
            # train_x, test_x, train_y, test_y = train_test_split(self._test_dataset.data, self._test_dataset.labels.ravel(), test_size=(1-options['test_ratio']))
            train_x, train_y = self._training_dataset.data, self._training_dataset.labels
            test_x, test_y = self._test_dataset.data, self._test_dataset.labels
            train_y.resize(train_y.shape[0])
            test_y.resize(test_y.shape[0])
            dir_name = f'../result/'+ data_type +'/'
            if not os.path.exists(dir_name):
                os.makedirs(dir_name)
            
            if options['manually_one_vs_all'] and official_svm == False:
                # para_grid={'C':[1e-3, 1e-2, 5e-2],'gamma':[10, 15],'kernel':['poly']}
                # para_grid={'C':[0.1,5],'gamma':[10],'kernel':['poly','linear']}
                para_grid={'C':[0.1],'gamma':[10],'kernel':['poly']}
                grid = GridSearchCV(SVC(probability=True),para_grid,refit=True,verbose=0)
                grid.fit(train_x,train_y)
                # print(grid.best_estimator_)
                grid_predictions=grid.predict(test_x)
                metrics=classification_report(test_y,grid_predictions,output_dict=True)
                # print(metrics)
                #save the confusion matrix 
                disp=ConfusionMatrixDisplay.from_estimator(
                    grid,test_x,test_y,display_labels=[-1,1],cmap=plt.cm.Blues)
                disp.ax_.set_title("Confusion Matrix for" + " Class "+str(options['num']))
                plt.savefig(dir_name+str(options['num'])+"_cfm.jpg")
                plt.close()
                
                """
                file_path=dir_name+str(options['num'])+'model.pickle'
                with open(file_path,'wb') as fp:
                    pickle.dump(grid,fp)
                np.save(dir_name+str(options['num'])+'classification_report.npy',classification_report(test_y,grid_predictions))
                """

                acc_score=accuracy_score(test_y,grid_predictions)

                #plot the FP and FN 
                FP=np.where((test_y==-1)&(grid_predictions==1))[0]
                FN=np.where((test_y==1)&(grid_predictions==-1))[0]
                fig= plt.figure(figsize=(8,8))
                plt.title(data_type+" class "+str(class_num)+" false result ",pad=30,fontsize=15)

                for i in range(0,4):
                    if i < 2:
                        if i >= len(FP):
                            continue
                        idx=FP[i]
                        title="False Positive"
                    else:
                        if i >= len(FN):
                            continue
                        idx=FN[i-2]
                        title="False Negative"
                    fig.add_subplot(2,2,i+1)

                    img_mat = test_x[idx]
                    img_length = int(np.sqrt(self._training_dataset.data_dim))
                    img=img_mat.reshape((img_length, img_length))
                    plt.imshow(img)
                    plt.title(title)
                #
                plt.savefig(dir_name+str(options['num'])+"_misclassified.jpg")
                plt.close()
                
                grid_prob = grid.predict_proba(self._test_dataset.data).copy()
                return acc_score, metrics, grid_prob
               
            elif options['official_svm'] and official_svm ==True:
                if options['OVO']:
                    #Since ovo will cause 10*9/2=45 classifers, if we use cross-validation, it will cause a lot. 
                    auto_svm(train_x,train_y,test_x,test_y,dir_name,svm_type='ovo',options=options)
                if options['OVR']:
                    auto_svm(train_x,train_y,test_x,test_y,dir_name,svm_type='ovr',options=options)
                return 

                
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


        


def auto_svm(train_x,train_y,test_x,test_y,dir_name,svm_type,options=options):
    """
    Call sklearn SVM api.

    param svm_type: indicate one-vs-one(ovo) or one-vs-rest(ovr)    
    """

    #Since ovo will cause 10*9/2=45 classifers, if we use cross-validation, it will cause a lot. 
    clf = SVC(decision_function_shape=svm_type,C=options['C'],gamma=options['gamma'],kernel=options['kernel'])
    clf.fit(train_x,train_y)
    ovo_prediction=clf.predict(test_x)
    acc=accuracy_score(test_y,ovo_prediction)
    print("Non-manual "+svm_type+" SVM accuracy is:",acc)
    disp=ConfusionMatrixDisplay.from_estimator(
    clf,test_x,test_y,display_labels=range(0,10),cmap=plt.cm.Blues)
    disp.ax_.set_title("Confusion Matrix for " +svm_type+", and accuracy is: "+ str(acc))
    plt.savefig(dir_name+svm_type+"_cfm.jpg")
    plt.close()



def experiment(data_type,feature_extraction,method=None,official_svm=False,options=options, feature_alg_opt=None):

    """
    The experiment function is divided into two parts.

    The first part is our hand-written one-to-rest classifier. 

    In the second part, we call the sklearn svm api to do the multi-class classification.
    (ps: the underlying logic of it is also one-vs-one or one-vs-rest classifier.)

    :param data_type: str, describe the data type(image or feature) and the SVM type we are running.
    :param feature_extraction: bool, indicate whether use feature extraction method or not.
    :param method:str, feature extraction method. 'pca','scale','mixed'
    :param official svm:bool, indicate use manually one-vs-rest svm or the multi-class svm api.

    """
    t0 = time.time()
    dir_name = f'../result/'+ data_type+'/'
    if official_svm==False:
        acc_all=[]
        precision_all=[]
        recall_all=[]
        grid_probs = []

        for i in range(options['class_num']):
            options['num']=i
            dataset_for_digit = dataset.sub_dataset(i)
            test_dataset_for_digit = test_dataset.sub_dataset(i)
            classifier = ClassificationModel(dataset_for_digit, test_dataset_for_digit)
            #classifier.save('digit-'+str(i))
            if feature_extraction == True:
                feature_alg_opt = {
                    'scale_ratio': .9,
                    'resampling_alg': 'bilinear',
                    'dim_output': 3 * 3
                } if feature_alg_opt is None else feature_alg_opt
                classifier.extract_feature(method=method, save_to_dataset=True, options=feature_alg_opt)

            acc, metrics, grid_prob=classifier.learn(data_type,official_svm=official_svm,alg='SVM',options=options)
            acc_all.append(acc)
            precision_all.append(metrics['1.0']['precision'])
            recall_all.append(metrics['1.0']['recall'])
            grid_probs.append(grid_prob)

        # Overall classification test
        right_counter = 0
        for i in range(len(test_dataset.data)):
            predict_prob = [grid_prob[i][1] for grid_prob in grid_probs]
            predicted_num = np.argmax(predict_prob)
            if predicted_num == test_dataset.labels[i][0]:
                right_counter += 1
        print(f'Overall accuracy: {round(right_counter / len(test_dataset.data), 5)}')

        x_axis=range(0,options['class_num'])
        plt.plot(x_axis,acc_all,label='SVM accuracy')
        plt.xlabel('class')
        plt.ylabel('accuracy')
        for a, b in zip(x_axis,acc_all):
            b=round(b,3)
            plt.text(a,b,b,ha='center',va='bottom',fontsize=10)
        plt.legend()
        plt.title(data_type+" Accuracy ")
        plt.savefig(dir_name+'accuracy_svm.jpg',dpi=300)
        #plt.show()
        plt.close()

        plt.plot(x_axis,precision_all,label='SVM precision')
        plt.legend()
        plt.plot(x_axis,recall_all,label='SVM recall')
        plt.legend()
        plt.title(data_type+" Precision and Recall ")
        plt.savefig(dir_name+'precision_recall.jpg',dpi=300)
        plt.close()
    elif official_svm:
        classifier=ClassificationModel(dataset, test_dataset)
        if feature_extraction == True:
            classifier.extract_feature(method=method, save_to_dataset=True, options={
                    'scale_ratio': .9,
                    'resampling_alg': 'bilinear',
                    'dim_output': 3 * 3
                    })
        #classifier.extract_feature(method='scale', save_to_dataset=True, options={'scale_ratio': .2})
        acc=classifier.learn(data_type,official_svm=True,alg='SVM', options=options)
    print(f'''{data_type}: overall time is {time.time() - t0} seconds''')

dataset, test_dataset = None, None
first_half_as_training = True

if __name__ == '__main__':
    data1, data2 = np.split(Dataset.load_matrix('../data/digits4000_digits_vec.txt'), 2, axis=0)
    label1, label2 = np.split(Dataset.load_matrix('../data/digits4000_digits_labels.txt'), 2, axis=0)
    dataset = Dataset(data1, label1) if first_half_as_training else Dataset(data2, label2)
    test_dataset = Dataset(data2, label2) if first_half_as_training else Dataset(data1, label1)

    """
    fig, axes = plt.subplots(nrows=2)
    classifier = ClassificationModel(dataset)
    idx = classifier.show_im(ax=axes[0])
    
    classifier.extract_feature(method='pca', save_to_dataset=True, options={
    'scale_ratio': .9,
    'resampling_alg': 'bilinear',
    'dim_output': 3 * 3
        })
    
    
    classifier = ClassificationModel(dataset)
    """

    for pca_output_dim in [20, 14, 8]:
        prescale_ratio = (28 + pca_output_dim) / 2 / 28

        experiment(f'''PCA-to-{pca_output_dim}-with-Prescaling''', feature_extraction=True, method='mixed', feature_alg_opt={
            'scale_ratio': prescale_ratio,
            'resampling_alg': 'bilinear',
            'dim_output': pca_output_dim * pca_output_dim
        })

        experiment(f'''PCA-to-{pca_output_dim}''', feature_extraction=True, method='pca', feature_alg_opt={
            'dim_output': pca_output_dim * pca_output_dim
        })

        experiment(f'''Pure-Scale-to-{pca_output_dim}''', feature_extraction=True, method='scale', feature_alg_opt={
            'scale_ratio': pca_output_dim / 28
        })



    '''
    #comment the part you donnot need. Otherwise it is time consuming.

    #use image 
    if options['manually_one_vs_all']:
        experiment("manually_ovr_img",feature_extraction=False) #our one to rest classifier
    if options['official_svm']:
        experiment("official_svm_img",feature_extraction=False,official_svm=True) #sklearn multiclass api

    #use pca 
    if options['extract_feature'] and options['pca']:
        if options['manually_one_vs_all']:      
            experiment("manually_ovr_pca",feature_extraction=True,method='pca')
        if options['official_svm']:
           experiment("official_svm_pca",feature_extraction=True,method='pca',official_svm=True) 
    #use scale
    if options['extract_feature'] and options['scale']:
        if options['manually_one_vs_all']:        
            experiment("manually_ovr_scale",feature_extraction=True,method='scale')
        if options['official_svm']:
            experiment("official_svm_scale",feature_extraction=True,method='scale',official_svm=True) 

    #use mixed
    if options['extract_feature'] and options['mixed']:
        if options['manually_one_vs_all']:       
            experiment("manually_ovr_mixed",feature_extraction=True,method='mixed')
        if options['official_svm']:
            experiment("official_svm_mixed",feature_extraction=True,method='mixed',official_svm=True) 
    '''

    """
    classifier.show_im(idx, ax=axes[1])
    dir_name = f'../saved_model/'
    plt.savefig(dir_name+"feature_img"+str(idx)+"_.jpg")
    plt.show()



    experiment("test",feature_extraction=False,official_svm=False)
    """


    # print(classifier.extract_feature(method='PCA', options={'dim_output': 15}))

