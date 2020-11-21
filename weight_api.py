import warnings
from collections import OrderedDict
from functools import wraps
from inspect import signature, Parameter
from numbers import Integral, Real
import numpy as np
from sklearn.base import clone
from sklearn.neighbors._base import KNeighborsMixin
from sklearn.neighbors import NearestNeighbors
from sklearn.utils import column_or_1d
from sklearn.utils.multiclass import type_of_target
from imblearn.exceptions import raise_isinstance_error





def check_neighbors_object(nn_name, nn_object, additional_neighbor=0):
    """Check the objects is consistent to be a NN.

    Several methods in imblearn relies on NN. Until version 0.4, these
    objects can be passed at initialisation as an integer or a
    KNeighborsMixin. After only KNeighborsMixin will be accepted. This
    utility allows for type checking and raise if the type is wrong.

    Parameters
    ----------
    nn_name : str,
        The name associated to the object to raise an error if needed.

    nn_object : int or KNeighborsMixin,
        The object to be checked

    additional_neighbor : int, optional (default=0)
        Sometimes, some algorithm need an additional neighbors.

    Returns
    -------
    nn_object : KNeighborsMixin
        The k-NN object.
    """
    if isinstance(nn_object, Integral):
        return NearestNeighbors(n_neighbors=nn_object + additional_neighbor)
    elif isinstance(nn_object, KNeighborsMixin):
        return clone(nn_object)
    else:
        raise_isinstance_error(nn_name, [int, KNeighborsMixin], nn_object)





    def _in_danger_noise(
        self, nn_estimator, samples, target_class, y, kind="danger"
    ):
        """Estimate if a set of sample are in danger or noise.

        Used by BorderlineSMOTE and SVMSMOTE.

        Parameters
        ----------
        nn_estimator : estimator
            An estimator that inherits from
            :class:`sklearn.neighbors.base.KNeighborsMixin` use to determine if
            a sample is in danger/noise.

        samples : {array-like, sparse matrix} of shape (n_samples, n_features)
            The samples to check if either they are in danger or not.

        target_class : int or str
            The target corresponding class being over-sampled.

        y : array-like of shape (n_samples,)
            The true label in order to check the neighbour labels.

        kind : {'danger', 'noise'}, default='danger'
            The type of classification to use. Can be either:

            - If 'danger', check if samples are in danger,
            - If 'noise', check if samples are noise.

        Returns
        -------
        output : ndarray of shape (n_samples,)
            A boolean array where True refer to samples in danger or noise.
        """
        x = nn_estimator.kneighbors(samples, return_distance=False)[:, 1:]
        # print('X:\t',x)
        nn_label = (y[x] != target_class).astype(int)
        n_maj = np.sum(nn_label, axis=1)
        # print('每个少数类点K近邻中多数类的个数:\t',n_maj,len(n_maj))
        # print('nn_estimator.n_neighbors:\t',nn_estimator.n_neighbors)     # ==self.k_neigbors

        if kind == "danger":
            # Samples are in danger for m/2 <= m' < m
            return np.bitwise_and(          #这里-1的原因是模型初始化的时候+1了
                n_maj >= (nn_estimator.n_neighbors - 1) / 2,
                n_maj < nn_estimator.n_neighbors - 1,
            ),n_maj

        elif kind == "noise":
            # Samples are noise for m = m'
            return n_maj == nn_estimator.n_neighbors - 1,n_maj
        else:
            raise NotImplementedError



    def _in_danger_noise(
        self, nn_estimator, samples, target_class, y, kind="danger"
    ):
        """Estimate if a set of sample are in danger or noise.

        Used by BorderlineSMOTE and SVMSMOTE.

        Parameters
        ----------
        nn_estimator : estimator
            An estimator that inherits from
            :class:`sklearn.neighbors.base.KNeighborsMixin` use to determine if
            a sample is in danger/noise.

        samples : {array-like, sparse matrix} of shape (n_samples, n_features)
            The samples to check if either they are in danger or not.

        target_class : int or str
            The target corresponding class being over-sampled.

        y : array-like of shape (n_samples,)
            The true label in order to check the neighbour labels.

        kind : {'danger', 'noise'}, default='danger'
            The type of classification to use. Can be either:

            - If 'danger', check if samples are in danger,
            - If 'noise', check if samples are noise.

        Returns
        -------
        output : ndarray of shape (n_samples,)
            A boolean array where True refer to samples in danger or noise.
        """
        x = nn_estimator.kneighbors(samples, return_distance=False)[:, 1:]
        # print('X:\t',x)
        nn_label = (y[x] != target_class).astype(int)
        n_maj = np.sum(nn_label, axis=1)
        # print('每个少数类点K近邻中多数类的个数:\t',n_maj,len(n_maj))
        # print('nn_estimator.n_neighbors:\t',nn_estimator.n_neighbors)     # ==self.k_neigbors

        if kind == "danger":
            # Samples are in danger for m/2 <= m' < m
            return np.bitwise_and(          #这里-1的原因是模型初始化的时候+1了
                n_maj >= (nn_estimator.n_neighbors - 1) / 2,
                n_maj < nn_estimator.n_neighbors - 1,
            ),n_maj

        elif kind == "noise":
            # Samples are noise for m = m'
            return n_maj == nn_estimator.n_neighbors - 1,n_maj
        else:
            raise NotImplementedError



def in_danger_noise(
        nn_estimator, samples, target_class, y, kind="danger"
    ):
        x = nn_estimator.kneighbors(samples, return_distance=False)[:, 1:]
        # print('X:\t',x)
        nn_label = (y[x] != target_class).astype(int)
        n_maj = np.sum(nn_label, axis=1)
        # print('每个少数类点K近邻中多数类的个数:\t',n_maj,len(n_maj))
        # print('nn_estimator.n_neighbors:\t',nn_estimator.n_neighbors)     # ==self.k_neigbors

        if kind == "danger":
            # Samples are in danger for m/2 <= m' < m
            return np.bitwise_and(          #这里-1的原因是模型初始化的时候+1了
                n_maj >= (nn_estimator.n_neighbors - 1) / 2,
                n_maj < nn_estimator.n_neighbors - 1,
            ),n_maj

        elif kind == "noise":
            # Samples are noise for m = m'
            return n_maj == nn_estimator.n_neighbors - 1,n_maj
        else:raise NotImplementedError






def add_weight(X,y,X_min,minority_label,
    base_indices,neighbor_indices,num_to_sample,
    ind,X_neighbor,X_base):
    from weight_api import check_neighbors_object,in_danger_noise
    import random

    nn_m_ = check_neighbors_object(
        "m_neighbors", 5, additional_neighbor=1     #TODO
    )
    nn_m_.set_params(**{"n_jobs": 1})
    nn_m_.fit(X)       #在所有点中求少数点的近邻点，以此来求少数点的权重
    noise,n_maj = in_danger_noise(
        nn_m_, X_min, minority_label, y, kind="noise" 
        )
    
    def conut_weight(n_maj): 
        # new_n_maj = [round((1-i/5),2) for i in n_maj]
        return [round((1-i/5),2) for i in n_maj]
    new_n_maj = np.array(conut_weight(n_maj=n_maj))
    
    X_base_weight = new_n_maj[base_indices]
    X_neighbor_weight = new_n_maj[ind[base_indices,neighbor_indices]]
    
    weights = []
    delete_index = []
    for n in range(int(num_to_sample)):


        if X_base_weight[n]!=0 and X_neighbor_weight[n]!=0: #如果母点和随机点权重都不是噪声点
            if X_base_weight[n]>= X_neighbor_weight[n]:
                proportion = (X_neighbor_weight[n] / (X_base_weight[n]+X_neighbor_weight[n])*round(random.uniform(0,1),len(str(num_to_sample))))#权重比例
            elif X_base_weight[n]< X_neighbor_weight[n]:
                proportion = X_neighbor_weight[n] / (X_base_weight[n]+X_neighbor_weight[n])
                proportion = proportion+(1-proportion)*(round(random.uniform(0,1),len(str(num_to_sample))))#权重比例
        elif (X_base_weight[n]+X_neighbor_weight[n])==0:       #如果母点和随机点权重都是0（两个点都是噪声点）
            delete_index.append(n)
            continue       
        elif X_base_weight[n] ==0 and X_neighbor_weight[n]!=0:
            proportion = 1
        elif X_base_weight[n] !=0 and X_neighbor_weight[n] ==0:
            proportion = 0
        
  
        weights.append(proportion)
    
    X_neighbor = np.delete(X_neighbor,delete_index,axis=0)
    X_base = np.delete(X_base,delete_index,axis=0)
    # print(X_base.shape)

    # weights=np.array(weights).reshape(int(num_to_sample),1)
    weights=np.array(weights).reshape(int(len(weights)),1)
    # print(weights.shape)

    samples= X_base + np.multiply(weights, X_neighbor - X_base)
    return samples



