import os
import warnings
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.datasets import (make_blobs, make_circles, make_classification,
                            make_moons)
from imblearn import over_sampling
import all_smote_v2
import imbalanced_databases as imbd
import minisom



def fourclass_data():
    data = pd.read_csv(r'fourclass10_change.csv', header=None)
    # data = pd.read_csv(r'iris.csv', header=None)
    y = data[0]
    X = data[[1, 2]]
    X = X.values
    y = y.values
    # print(X,y)

    return X, y


def load_data(data_name, random_state=0):
    np.random.seed(random_state)
    if data_name == 'make_moons':
        x, y = make_moons(n_samples=1200, noise=0.3)
        data = np.hstack((y.reshape((len(y), 1)), x))
        np.random.shuffle(data)
        # print(Counter(data[:, 0]))

        data_0 = data[data[:, 0] == 0]
        data_1 = data[data[:, 0] == 1]

        data_0 = data_0[:100]

        data = np.vstack((data_0, data_1))
        # print(type(data))

    elif data_name == 'make_circles':
        x, y = make_circles(n_samples=1200, noise=0.1, factor=0.4)
        data = np.hstack((y.reshape((len(y), 1)), x))
        np.random.shuffle(data)
        # print(Counter(data[:, 0]))

        data_0 = data[data[:, 0] == 0]
        data_1 = data[data[:, 0] == 1]

        data_0 = data_0[:100]

        data = np.vstack((data_0, data_1))
    
    data = pd.DataFrame(data)
    # print(data)
    y = data[0]
    X = data[[1, 2]]
    X = X.values
    y = y.values
    return X, y


def main(X=None, y=None):  # ndarray

    fig, axes = plt.subplots(4, 2, figsize=(10, 10))
    X = pd.DataFrame(X)
    print('\nOriginal dataset shape %s' % Counter(y), '\n')
    font2 = {'family': 'Times New Roman',
                'size': 20, }




    '''weight-smote'''
    sm = all_smote_v2.SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X, y)
    print('weight smote %s' % Counter(y_res), '\n\n\n')
    y_res = pd.DataFrame(y_res)
    X_new = X_res.iloc[len(X):, :]
    y_new = y_res.iloc[len(y):, :]  
    for i in range(len(X)):
        if y[i] == 1:
            # print(X.iloc[i][0])
            axes[0][0].scatter(X.iloc[i][0], X.iloc[i][1], c='tan', s=25)
    for i in range(len(X)):
        if y[i] == 0:
            axes[0][0].scatter(X.iloc[i][0], X.iloc[i][1], c='darkcyan', s=25)
    axes[0][0].scatter(X_new[0], X_new[1], c='red', marker='+', s=50)
    axes[0][0].set_title('(a) weight-SMOTE ', font2)



    '''smote'''
    sm = over_sampling.SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X, y)
    print('smote %s' % Counter(y_res), '\n\n\n')
    y_res = pd.DataFrame(y_res)
    X_new = X_res.iloc[len(X):, :]
    y_new = y_res.iloc[len(y):, :]  
    for i in range(len(X)):
        if y[i] == 1:
            # print(X.iloc[i][0])
            axes[0][1].scatter(X.iloc[i][0], X.iloc[i][1], c='tan', s=25)
    for i in range(len(X)):
        if y[i] == 0:
            axes[0][1].scatter(X.iloc[i][0], X.iloc[i][1], c='darkcyan', s=25)
    axes[0][1].scatter(X_new[0], X_new[1], c='red', marker='+', s=50)
    axes[0][1].set_title('(b) SMOTE ', font2)


    '''boderline_1'''
    sm_1 = over_sampling.BorderlineSMOTE(random_state=42, kind="borderline-1")
    X_res, y_res = sm_1.fit_resample(X, y)
    print('borderline-1 shape %s' % Counter(y_res), '\n\n\n\n')
    y_res = pd.DataFrame(y_res)
    X_new = X_res.iloc[len(X):, :]
    y_new = y_res.iloc[len(y):, :]
    for i in range(len(X)):
        if y[i] == 1:
            # print(X.iloc[i][0])
            axes[1][1].scatter(X.iloc[i][0], X.iloc[i][1], c='tan', s=25)
    for i in range(len(X)):
        if y[i] == 0:
            axes[1][1].scatter(X.iloc[i][0], X.iloc[i][1], c='darkcyan', s=25)
            # axes[0][0].scatter(X.iloc[i:,0:], X[i:,1:], c='cyan',)
    axes[1][1].scatter(X_new[0], X_new[1], c='red', marker='+', s=50)
    axes[1][1].set_title('(d) boderline-SMOTE1', font2)




    '''weight-boderline'''
    sm_zhou = all_smote_v2.BorderlineSMOTE(random_state=42, kind="weight-borderline-smote", )
    X_res, y_res = sm_zhou.fit_resample(X, y)
    print('weight-borderline shape %s' % Counter(y_res), '\n\n\n')
    y_res = pd.DataFrame(y_res)
    X_new = X_res.iloc[len(X):, :]
    y_new = y_res.iloc[len(y):, :]
    for i in range(len(X)):
        if y[i] == 1:
            # print(X.iloc[i][0])
            axes[1][0].scatter(X.iloc[i][0], X.iloc[i][1], c='tan', s=25)
    for i in range(len(X)):
        if y[i] == 0:
            axes[1][0].scatter(X.iloc[i][0], X.iloc[i][1], c='darkcyan', s=25)
            # axes[0][0].scatter(X.iloc[i:,0:], X[i:,1:], c='cyan',)
    axes[1][0].scatter(X_new[0], X_new[1], c='red', marker='+', s=50)
    axes[1][0].set_title('(c) weight-boderline', font2)
    
    
    '''weight-kmeans-smote'''
    sm_3 = all_smote_v2.KMeansSMOTE(random_state=42, kind='kmeans-borderline')
    X_res, y_res = sm_3.fit_resample(X, y)
    print('weight-kmeans-smote:\t', Counter(y_res), '\n\n\n\n')
    y_res = pd.DataFrame(y_res)
    X_new = X_res.iloc[len(X):, :]
    y_new = y_res.iloc[len(y):, :]
    axes[2][0].scatter(X[0], X[1], c=y, alpha=0.5)
    axes[2][0].scatter(X_new[0], X_new[1], c='red', alpha=0.2)
    axes[2][0].set_title('weight_kmeans_smote')
    
    
    
    
    
    
    '''kmeans-smote'''
    sm_4 = over_sampling.KMeansSMOTE(random_state=42,)
    X_res, y_res = sm_4.fit_resample(X, y)
    print('kmeans-smote:\t', Counter(y_res), '\n\n\n\n')
    y_res = pd.DataFrame(y_res)
    X_new = X_res.iloc[len(X):, :]
    y_new = y_res.iloc[len(y):, :]
    axes[2][1].scatter(X[0], X[1], c=y, alpha=0.5)
    axes[2][1].scatter(X_new[0], X_new[1], c='red', alpha=0.2)
    axes[2][1].set_title('kmeans_smote')


    


    '''weight-svm-smote'''
    sm_7 = all_smote_v2.SVMSMOTE(random_state=42,)
    X_res, y_res = sm_7.fit_resample(X, y)
    print('weight-SVM_smote:\t', Counter(y_res),'\n\n\n\n')
    y_res = pd.DataFrame(y_res)
    X_new = X_res.iloc[len(X):, :]
    y_new = y_res.iloc[len(y):, :]
    axes[3][0].scatter(X[0], X[1], c=y, alpha=0.5)
    axes[3][0].scatter(X_new[0], X_new[1], c='red', alpha=0.2)
    axes[3][0].set_title('weight_svm_smote')



    '''SVM_SMOTE'''
    sm_6 = over_sampling.SVMSMOTE(random_state=42,)
    X_res, y_res = sm_6.fit_resample(X, y)
    print('SVM_smote:\t', Counter(y_res),'\n\n\n\n')
    y_res = pd.DataFrame(y_res)
    X_new = X_res.iloc[len(X):, :]
    y_new = y_res.iloc[len(y):, :]
    axes[3][1].scatter(X[0], X[1], c=y, alpha=0.5)
    axes[3][1].scatter(X_new[0], X_new[1], c='red', alpha=0.2)
    axes[3][1].set_title('svm_smote')



def plot(X,y):
    plt.figure(figsize=(10,10))
    X = pd.DataFrame(X)
    print('\nOriginal dataset shape %s' % Counter(y), '\n')
    font2 = {'family': 'Times New Roman',
                'size': 20, }
    for i in range(len(X)):
        if y[i] == 1:
            # print(X.iloc[i][0])
            plt.scatter(X.iloc[i][0], X.iloc[i][1], c='tan', s=25)
    for i in range(len(X)):
        if y[i] == 0:
            plt.scatter(X.iloc[i][0], X.iloc[i][1], c='darkcyan', s=25)
    
    # plt.savefig(fname='original.pdf',format='pdf',bbox_inches='tight')

if __name__ == "__main__":
    # X,y = fourclass_data()
    X, y = load_data(data_name='make_moons')
    X, y = load_data(data_name='make_circles')

    main(X, y)
    # plot(X,y)

    plt.tight_layout()
    # plt.savefig(fname='pic.pdf',format='pdf',bbox_inches='tight')
    plt.show()
    print('运行完毕！')
