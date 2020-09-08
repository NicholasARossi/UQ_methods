import numpy as np
from sklearn.model_selection import KFold,train_test_split,LeaveOneOut,ShuffleSplit
from sklearn.linear_model import LinearRegression
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from scipy.stats import spearmanr
import seaborn as sns
from easing import easing
import pandas as pd
import os
from sampling_methods import continuous_stratification,stratify_all


class EnsembleModel():
    ''' This is the base class that enables the training of an ensemble model'''

    def __init__(self, X_data, y_data):
        self.X_train = X_data
        self.y_train = y_data

    def train_models(self, n_splits=5, kernal='linear',splitting='shuffle',one_sample_override=False):
        '''Function for training models
        Inputs:
        X_dataset: Inputs of the training data (MxN)
        y_dataset: Outputs of the trainig data (1xN)
        params: the parameters of the LGBM model
        n_split: the number of splits to be used to creat int(n_splits) seperate models
        kernal: the base model type for the commitee members
        splitting: random or stratified splitting

        Returns:
        models: int(n_splits) number of independent models trained on the folds of the data
        '''


        if splitting=='shuffle':
            splits = ShuffleSplit(test_size=1 / n_splits, n_splits=n_splits).split(self.X_train)

        if splitting=='kfold':
            splits= KFold(n_splits=n_splits).split(self.X_train)

        if splitting=='stratified':
            splits_list=stratify_all(self.y_train.ravel(),n_splits,n_bins=10,test_size=0.1,one_sample_override=one_sample_override)
            splits = (n for n in splits_list)

        if splitting not in ['shuffle','stratified','kfold']:
            print('splitting method not recognized')
            exit(1)
        #ss = ShuffleSplit(test_size=10, n_splits=n_splits, random_state=42)


        models = []
        coeffs = []
        for train_index, test_index in splits:
            X_train, X_val = self.X_train[train_index, :], self.X_train[test_index, :],
            y_train, y_val = self.y_train[train_index], self.y_train[test_index]

            if kernal=='linear':
                reg = LinearRegression(fit_intercept=False).fit(X_train, y_train)
                models.append(reg)
                coeffs.append(reg.coef_[0][0])

            if kernal=='LGB':
                params = {
                    'num_leaves': 50,
                    'metric': ['l1', 'l2'],
                    'verbose': -1,
                    'learning_rate': 1,
                    'lambda_l2': 0,
                    'drop_rate': 0,
                    'objective': 'regression'
                }
                lgb_train = lgb.Dataset(X_train, y_train.ravel())

                gbm = lgb.train(params,
                                lgb_train,
                                num_boost_round=800,
                                verbose_eval=0,
                                )

                models.append(gbm)




        self.models = models
        self.coefficients = coeffs

    def predict(self, X):
        ''' Predict method '''
        predictions = []
        for model in self.models:
            predictions.append(model.predict(X))

        # Values chosen by median committe
        ensemble_predictions = np.mean(np.stack(predictions, axis=0), axis=0)
        ensemble_uncertainty = np.std(np.stack(predictions, axis=0), axis=0)
        self.predictions = {'prediction': ensemble_predictions, 'uncertainty': ensemble_uncertainty}
        self.raw_predictions = {'X': np.squeeze(np.stack(predictions, axis=0))}

        return self.predictions

    def visualize_ensemble(self,X_holdout,destination='../Media',title='temp',animate=False):

        ''' Animation method for some of the above computation'''
        raw_pred = self.raw_predictions['X']
        dims = np.shape(raw_pred)

        output = self.predict(X_holdout)

        fig, ax = plt.subplots(2, 3, figsize=(12, 8), sharex=True, sharey=True, gridspec_kw={'hspace': 0.5})
        ax = ax.ravel()

        ax[0].set_xlim([0, 100])
        ax[0].set_ylim([0, 100])

        titles = ['Model 1', 'Model 2', 'Model 3', 'Model 4', 'Model 5','Ensemble']


        for j in range(5):
            sns.regplot(X_holdout.ravel(), raw_pred[j, :], ax=ax[j])

            ax[j].set_ylabel('Predicted Values')
            ax[j].set_title(f'{titles[j]} \n MSE: {mean_squared_error(X_holdout.ravel(), raw_pred[j, :]):.0f} \n spearman: {spearmanr(X_holdout.ravel(), raw_pred[j, :])[0]:.2f}')

        ensemble_predictions = output['prediction']
        ensemble_uncertainty = output['uncertainty']

        ax[5].set_title(f'{titles[j+1]} \n MSE: {mean_squared_error(X_holdout.ravel(), ensemble_predictions):.0f} \n spearman: {spearmanr(X_holdout.ravel(), ensemble_predictions)[0]:.2f}')
        ax[5].errorbar(X_holdout.ravel(), ensemble_predictions, yerr=ensemble_uncertainty, color='#905363', fmt='o')
        ax[5].plot([0, 100], [0, 50], color='#d3d3d3', linewidth=3)
        ax[5].set_ylabel('Predicted Values')
        fig.savefig(os.path.join(destination,f'{title}_multiplanel.png'),bbox_inches='tight')

        if animate==True:

            animation_dict_x_y={}


            for r in range(dims[0]):

                x=raw_pred[r,:]

                y=X_holdout.ravel()
                result = [None]*(len(x)+len(y))
                result[::2] = y
                result[1::2] = x
                animation_dict_x_y[f'Model {r+1}']=result

            easing.Eased(pd.DataFrame(animation_dict_x_y).T).scatter_animation2d(speed=0.5,
                                                                                 label=True,
                                                                                 destination=os.path.join(destination,f'{title}.gif'),
                                                                                 plot_kws={'alpha': 0.5,
                                                                                           'title': '5 Seperate Models',
                                                                                           'xlabel': 'Normalized Annual Income',
                                                                                           'ylabel': 'Predcited Values'})
if __name__ == '__main__':
    from synthetic_data_generator import meal_income

    meal_income_dict = meal_income(size=10000)
    x_values = meal_income_dict['X']
    y_values = meal_income_dict['y']

    X_dataset, X_holdout, y_dataset, y_holdout = train_test_split(x_values, y_values, test_size=0.1, random_state=42)

    LGBM_model = EnsembleModel(X_dataset, y_dataset)
    LGBM_model.train_models(n_splits=5, kernal='LGB',splitting='shuffle')
    LGBM_model.predict(X_holdout)
    LGBM_model.visualize_ensemble(X_holdout, destination='../media/',title='shuffle')

