import numpy as np
import pandas as pd

def meal_income(size=1000,return_df=False):
    x_values = np.sort(np.random.uniform(100, size=size))
    y_values = []
    for x_value in x_values:
        y_values.append(np.random.uniform(x_value))

    meal_dict={'X':x_values.reshape(-1,1),'y':np.array(y_values).reshape(-1,1),'mu':x_values/2,'sigma':np.sqrt(x_values**2/12)}

    if return_df==False:
        return meal_dict
    else:
        meal_income_df = pd.DataFrame()
        for key in meal_dict.keys():
            meal_income_df[key] = meal_dict[key].ravel()

        return meal_income_df

def two_feature_uncertainty(size=1000):

    df=pd.DataFrame(columns=['y','X0','X1'])

    y_values = np.sort(np.random.uniform(100, size=size))
    x1 = [np.random.uniform(l) for l in y_values]
    x0 = [100 - np.random.uniform(l) for l in y_values][::-1]

    df['y'] = y_values
    df['X0'] = x0
    df['X1'] = x1
    return df


def MLR(feature_noise_caller=False,output_noise_caller=True,n_features=10,data_size=101):
    '''This is a simple helper function that can be used to generate data that features
    heteroskedastic, aleatory uncertainty across multiple features that correlate linearly with an ouput
    PARAMS:


    '''

    x_features = [f'feature_{x + 1}' for x in range(n_features)]
    y_feature = ['output']

    all_features = x_features + y_feature
    pd.DataFrame(columns=all_features)
    outputs = np.arange(1, data_size)

    feature_dict = {'output': outputs}
    coeffs = []
    for j, feature in enumerate(x_features):
        coeff = np.random.normal(1)
        coeffs.append(coeff)
        if feature_noise_caller == True:
            feature_noise = np.random.normal(1, size=len(outputs))
            feature_dict[feature] = coeff * outputs + feature_noise
        else:
            feature_dict[feature] = coeff * outputs

    all_data = pd.DataFrame(feature_dict)
    if output_noise_caller == True:

        for idx, row in all_data.iterrows():
            all_data = all_data.append([row] * 9, ignore_index=True)
        all_data = all_data.sort_values(by='output')

        noise_vect = np.random.normal(scale=.01, size=len(all_data)) * np.arange(len(all_data))
        all_data['output'] += noise_vect
    all_data['ngroup'] = all_data.groupby(x_features).ngroup()
    X = all_data[x_features]
    y = all_data[y_feature]
    ngroups = X.groupby(list(X.columns)).ngroup()
    print(f'All Coefficients {coeffs}')

    return all_data, X, y, ngroups




