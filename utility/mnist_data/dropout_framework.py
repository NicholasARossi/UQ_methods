from sklearn.model_selection import KFold
from tensorflow.keras.models import Model
import numpy as np
import logging
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc

from tqdm import tqdm
from utility.mnist_data.model_warehouse import basic_CNN
from utility.mnist_data.adversarial_data import MNIST_EMNIST
import pandas as pd
from sklearn.metrics import confusion_matrix,accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
try:
    plt.style.use('rossidata')
except:
    sns.set_style("white")

class DropoutMinstModel:

    def __init__(self,base_model=basic_CNN,n_epochs=10,palette='flare'):
        self.base_model=base_model
        self.data=MNIST_EMNIST()


        ### training hyperparameters
        self.n_epochs=n_epochs
        self.metrics={}
        self.colors=sns.color_palette(palette, 3)
        self.palette=palette



    def _train_minst_model(self):
        self.trained_mnist_model = self.base_model()
        logging.info('Starting mnist model training')
        self.mnist_history = self.trained_mnist_model.fit(self.data.X_train, self.data.y_train, epochs=self.n_epochs, batch_size=32,
                                validation_data=(self.data.X_test, self.data.y_test), verbose=1)

        logging.info('mnist model trained')

        predictions = np.argmax(self.trained_mnist_model.predict(self.data.X_test), axis=-1)
        trues = self.data.revere_categorical(self.data.y_test)
        self.metrics['mnist_confusion_matrix']=confusion_matrix(trues, predictions)
        self.metrics['mnist_accuracy']=accuracy_score(trues,predictions)

    def _collect_dropout_uncertainty(self):
        logging.info('Collecting UQ from dropout')
        self.dropout_means, self.dropout_uncertainy = self.compute_dropout_uncertainty(self.trained_mnist_model, self.data.X_adv_train)

        self.uq_dataframe = pd.DataFrame({'UQ':self.dropout_uncertainy,
                               'is_adversarial':self.data.adversarial_training_labels,
                                          'Labels':self.data.holdout_training_keys})

        logging.info('Finished collecting UQ from dropout')




    def _train_dropout_uncertainty_model(self):
        self.uncertainty_model = LogisticRegression(random_state=0).fit(self.uq_dataframe['UQ'].values.reshape(-1, 1),
                                                                   self.uq_dataframe['is_adversarial'].values)

        predictions =  self.uncertainty_model.predict_proba(self.uq_dataframe['UQ'].values.reshape(-1, 1))[:, 1]
        fpr, tpr, _ = roc_curve(self.uq_dataframe['is_adversarial'].values, predictions)

        self.metrics['UQ_auc']=auc(fpr, tpr)
        self.metrics['UQ_fpr']=fpr
        self.metrics['UQ_tpr']=tpr




    def train(self):
        self._train_minst_model()
        self._collect_dropout_uncertainty()
        self._train_dropout_uncertainty_model()


    def make_plots(self,output_folder):

        ### uncerainty distributions
        fig, ax = plt.subplots()
        sns.boxplot(x='Labels', y='UQ', data=self.uq_dataframe, palette=self.palette, showfliers=False)

        ax.set_xticklabels([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 'A', 'C', 'E', 'F', 'H', 'K', 'M', 'N', 'R', 'X'])
        ax.set_title('Distributions of Uncertainty by class')
        fig.savefig(f'{output_folder}/uncertainty_distributions.png',dpi=300,bbox_inches='tight')

        ### Logistic regression
        fig, ax = plt.subplots()
        sns.regplot(x="UQ", y="is_adversarial", data=self.uq_dataframe,
                    logistic=True, n_boot=500, y_jitter=.1,color=self.colors[0],scatter_kws={'alpha':0.05},ax=ax)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['number', 'letter'])
        fig.savefig(f'{output_folder}/logistic_regression.png',dpi=300,bbox_inches='tight')


        ### MNIST ROC
        fig, ax = plt.subplots(figsize=(5,5))

        ax.plot(self.metrics['UQ_fpr'], self.metrics['UQ_tpr'],lw=3,color=self.colors[0])

        ax.plot([0, 1], [0, 1], color='#d3d3d3', linewidth=3)
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'Classifying Adversaries \n AUC :{np.round(self.metrics["UQ_auc"],2)}')
        fig.savefig(f'{output_folder}/roc_curve.png',dpi=300,bbox_inches='tight')

        ### confusion matrix
        fig, ax = plt.subplots()
        sns.heatmap(self.metrics['mnist_confusion_matrix'], annot=True,ax=ax)

    def predict(self,X):
        prediction_dict={}

        predict_dropout_mean, predict_dropout_uq = self.compute_dropout_uncertainty(self.trained_mnist_model, X)


        # get if adversarial
        prediction_dict['is_adversarial']=self.uncertainty_model.predict(X)


        # mnist prediction
        predicted_mnsist = self.trained_mnist_model.predict(X)
        predicted_classes = np.argmax(predicted_mnsist, axis=-1)

        prediction_dict['mnist_classes'] = predicted_classes
        prediction_dict['mnist_proba'] = predicted_mnsist

        return prediction_dict





    @staticmethod
    def compute_dropout_uncertainty(model,X,n_iter = 10):

        # f = K.function([model.layers[0].input, backend.symbolic_learning_phase()],[model.layers[-1].output])

        partial_model = Model(model.inputs, model.layers[-1].output)
        result = []

        for _ in tqdm(range(n_iter)):
            result.append(partial_model(X,training=True))

        result = np.array(result)

        prediction_mean = result.mean(axis=0)
        prediction_std = result.std(axis=0)
        return np.mean(prediction_mean,axis=1),np.mean(prediction_std,axis=1)