from sklearn.model_selection import KFold
from tensorflow.keras.models import Model
import numpy as np
import logging
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
import os
from tqdm import tqdm
from utility.mnist_data.model_warehouse import basic_CNN
from utility.mnist_data.adversarial_data import MNIST_EMNIST
import pandas as pd
from sklearn.metrics import confusion_matrix,accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

try:
    plt.style.use('rossidata')
except:
    sns.set_style("white")

class DropoutMinstModel:

    def __init__(self,base_model=basic_CNN,n_epochs=10,n_dropout_shuffles=10,palette='flare',title=None):
        self.base_model=base_model
        self.data=MNIST_EMNIST()

        self.title=title
        ### training hyperparameters
        self.n_epochs=n_epochs
        self.metrics={}
        self.colors=sns.color_palette(palette, 3)
        self.palette=palette
        self.n_dropout_shuffles=n_dropout_shuffles


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
        self.metrics['mnist_history']=self.mnist_history.history

    def _collect_dropout_uncertainty(self):
        logging.info('Collecting UQ from dropout')
        self.dropout_means, self.dropout_uncertainy = self.compute_dropout_uncertainty(self.trained_mnist_model,
                                                                                       self.data.X_adv_train,
                                                                                       n_iter=self.n_dropout_shuffles)

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

    def _serialize_outputs(self):
        if not os.path.exists(f'serialized_data/{self.title}'):
            os.makedirs(f'serialized_data/{self.title}')

        self.uq_dataframe.to_csv(f"serialized_data/{self.title}/uq_dataframe.csv")

        with open(f"serialized_data/{self.title}/metrics.pkl", 'wb') as handle:
            pickle.dump(self.metrics, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(f"serialized_data/{self.title}/prediction_dict.pkl", 'wb') as handle:
            pickle.dump(self.prediction_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

        self.trained_mnist_model.save(f'serialized_data/{self.title}/mnist.mdl')

        pickle.dump(self.uncertainty_model, open(f'serialized_data/{self.title}/uq.mdl', 'wb'))



    def train(self):
        self._train_minst_model()
        self._collect_dropout_uncertainty()
        self._train_dropout_uncertainty_model()
        self._test_on_adversarial()
        self._serialize_outputs()




    def make_plots(self,output_folder):
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        ### uncerainty distributions
        fig, ax = plt.subplots()
        sns.boxplot(x='Labels', y='UQ', data=self.uq_dataframe, palette=self.palette, showfliers=False)
        tickos=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 'A', 'C', 'E', 'F', 'K', 'M', 'N', 'R','V', 'X']
        ax.set_xticklabels(tickos)
        ax.set_title('Distributions of Uncertainty by class')
        fig.savefig(f'{output_folder}/uncertainty_distributions.png',dpi=300,bbox_inches='tight')

        ### Logistic regression
        fig, ax = plt.subplots()
        sns.regplot(x="UQ", y="is_adversarial", data=self.uq_dataframe,
                    logistic=True, n_boot=500, y_jitter=.2,line_kws={"color": self.colors[-1]},scatter_kws={'alpha':0.05,"color": self.colors[0]},ax=ax)
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




        ### Final Holdout ROC
        fig, ax = plt.subplots(figsize=(5,5))

        ax.plot(self.prediction_dict['fpr'], self.prediction_dict['tpr'],lw=3,color=self.colors[0])

        ax.plot([0, 1], [0, 1], color='#d3d3d3', linewidth=3)
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'Classifying True Holdout\n AUC :{np.round(self.prediction_dict["auc"],2)}')
        fig.savefig(f'{output_folder}/final_roc_curve.png',dpi=300,bbox_inches='tight')

        mnist_tickos=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

        ### confusion matrix
        fig, ax = plt.subplots()
        sns.heatmap(self.metrics['mnist_confusion_matrix'], annot=True,ax=ax,yticklabels=mnist_tickos,xticklabels=mnist_tickos)
        fig.savefig(f'{output_folder}/mnist_cm.png',dpi=300,bbox_inches='tight')


    def _test_on_adversarial(self):
        prediction_dict={}

        predict_dropout_mean, predict_dropout_uq = self.compute_dropout_uncertainty(self.trained_mnist_model,
                                                                                    self.data.X_adv_test,
                                                                                    n_iter=self.n_dropout_shuffles)



        # get if adversarial
        prediction_dict['is_adversarial']=self.uncertainty_model.predict(predict_dropout_uq.reshape(-1,1))
        prediction_dict['is_adversarial_proba']=self.uncertainty_model.predict_proba(predict_dropout_uq.reshape(-1,1))[:, 1]


        fpr, tpr, _ = roc_curve(self.data.adversarial_testing_labels, prediction_dict['is_adversarial_proba'])
        prediction_dict['fpr']=fpr
        prediction_dict['tpr']=tpr
        prediction_dict['auc']=auc(fpr, tpr)

        prediction_dict['adversarial_accuracy']=accuracy_score(self.data.adversarial_testing_labels,prediction_dict['is_adversarial'])

        # mnist prediction
        predicted_mnsist = self.trained_mnist_model.predict(self.data.X_adv_test)
        predicted_classes = np.argmax(predicted_mnsist, axis=-1)

        prediction_dict['mnist_classes'] = predicted_classes
        prediction_dict['mnist_proba'] = predicted_mnsist
        prediction_dict['true_mnist_classes']=self.data.holdout_testing_keys
        self.prediction_dict=prediction_dict




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


class KfoldMinstModel(DropoutMinstModel):
    def __init__(self,base_model=basic_CNN,n_epochs=10,n_folds=5,palette='flare'):
        super().__init__(base_model,n_epochs,palette)
        self.n_folds=n_folds

    def _train_minst_kfold_model(self):
        kfold = KFold(self.n_folds, shuffle=True, random_state=88)
        fold=1
        self.models=[]
        for train_ix, test_ix in kfold.split(self.data.X_train):
            trainX, trainY, testX, testY = self.data.X_train[train_ix], self.data.y_train[train_ix], self.data.X_train[test_ix], self.data.y_train[test_ix]

            trained_mnist_model = self.base_model()

            self.mnist_history = trained_mnist_model.fit(trainX, trainY, epochs=self.n_epochs, batch_size=32,
                                    validation_data=(testX, testY), verbose=0)
            #
            logging.info(f'mnist model trained :fold {fold}')
            self.models.append(trained_mnist_model)
            fold+=1


    def _test_minst_kfold(self):



        values,uncertainty=self.trained_mnist_model.predict(self.data.X_test)
        predictions = np.argmax(values, axis=-1)
        trues = self.data.revere_categorical(self.data.y_test)
        self.metrics['mnist_confusion_matrix']=confusion_matrix(trues, predictions)
        self.metrics['mnist_accuracy']=accuracy_score(trues,predictions)

    def _collect_kfold_uncertainty(self):
        logging.info('Collecting UQ from dropout')
        self.kfold_means, self.kfold_uncertainty = self.compute_kfold_uncertainty(self.models, self.data.X_adv_train)

        self.uq_dataframe = pd.DataFrame({'UQ':self.kfold_uncertainty,
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
        self._train_minst_kfold_model()
        self._collect_kfold_uncertainty()
        self._train_dropout_uncertainty_model()

    def predict(self,X):
        prediction_dict={}

        predict_kfold_mean, predict_kfold_uq = self.compute_kfold_uncertainty(self.models, X)



        # get if adversarial
        prediction_dict['is_adversarial']=self.uncertainty_model.predict(predict_kfold_uq.reshape(-1,1))
        prediction_dict['is_adversarial_proba']=self.uncertainty_model.predict_proba(predict_kfold_uq.reshape(-1,1))[:, 1]


        fpr, tpr, _ = roc_curve(self.data.adversarial_testing_labels, prediction_dict['is_adversarial_proba'])
        prediction_dict['fpr']=fpr
        prediction_dict['tpr']=tpr
        prediction_dict['auc']=auc(fpr, tpr)


        # mnist prediction
        predicted_mnsist,_ = self.compute_kfold_uncertainty(self.models,X)
        predicted_classes = np.argmax(predicted_mnsist, axis=-1)

        prediction_dict['mnist_classes'] = predicted_classes
        prediction_dict['mnist_proba'] = predicted_mnsist

        return prediction_dict

    @staticmethod
    def compute_kfold_uncertainty(models,X):

        predictions =[]
        for model in models:
            predictions.append(model.predict(X))
        means=np.mean(np.stack(predictions, axis=0) ,axis=0)
        uncertainties=np.std(np.stack(predictions, axis=0) ,axis=0)

        return np.mean(means,axis=1),np.mean(uncertainties,axis=1)



