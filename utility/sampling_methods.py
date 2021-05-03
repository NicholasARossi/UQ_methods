# create stratified holdout set
import numpy as np


def continuous_stratification(y,n_bins,test_size=0.1,one_sample_override=False):
    '''

    :param y: array to stratify on
    :param n_bins: number of bins to group y by
    :param test_size: total fraction of samples to return
    :return:
    '''
    '''Function that returns the stratified indexes of a continously valued y'''
    bins=np.linspace(min(y)-1, max(y)+1,n_bins)
    digitized_y=np.digitize(y,bins)
    index_list=np.arange(len(y))
    stratified_indexes=[]

    n_samples=int(len(y)*test_size)
    bucket_list=list(set(digitized_y))
    n_samples_per_bucket=int(n_samples/len(bucket_list))
    if one_sample_override==False:
        for bucket in list(set(digitized_y)):
            sub_index_list=index_list[digitized_y==bucket]
            if len(sub_index_list)>=n_samples_per_bucket:
                sample_indexes=np.random.choice(sub_index_list,size=n_samples_per_bucket,replace=False)
                stratified_indexes+=list(sample_indexes)
            else:
                print('insufficient samples in bin, decrease n_bins or test_size')
                stratified_indexes += list(sub_index_list)
    elif one_sample_override==True:
        for bucket in list(set(digitized_y)):
            sub_index_list=index_list[digitized_y==bucket]

            sample_indexes=np.random.choice(sub_index_list,size=1,replace=False)
            stratified_indexes+=list(sample_indexes)

    # returns test and train indexes
    return np.array(stratified_indexes),np.setdiff1d(index_list,stratified_indexes)


def stratify_all(y,n_splits,n_bins=20,test_size=0.1,one_sample_override=False):
    stratified_pairs=[]
    for n in range(n_splits):
        test, train = continuous_stratification(y, n_bins, test_size=test_size,one_sample_override=one_sample_override)
        stratified_pairs.append([test,train])
    return stratified_pairs

