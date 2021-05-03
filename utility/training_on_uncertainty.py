from sklearn.decomposition import PCA

import numpy as np


def PCA_binning(data,x_cols=None,y_col=None,n_components=1,n_bins=100):
    '''

    :param data:
    :param x_cols:
    :param y_cols:
    :return:
    '''
    pca = PCA(n_components=n_components)
    data['PCA_values'] = pca.fit_transform(data[x_cols])

    bins = np.linspace(min(data['PCA_values']), max(data['PCA_values']), n_bins)
    data['bins'] = np.digitize(data['PCA_values'], bins=bins, right=True)

    reduced_df=data.groupby('bins').mean().rename(columns={y_col:'y_mean'})
    reduced_df.loc[:,'y_std']=data.groupby('bins').std().rename(columns={y_col:'y_std'})['y_std']

    return reduced_df,data



