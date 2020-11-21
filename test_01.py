import numpy as np
import matplotlib.pyplot as plt
import _smote_variants_v2 as sv2    #加权
import  _smote_variants_v1 as sv1   #原始
import imbalanced_databases as imbd
import warnings
warnings.filterwarnings("ignore")
from boderline_zhou import load_data,fourclass_data
import pandas as pd
import time
from imblearn import over_sampling
import all_smote_v2,all_smote_v1
import minisom
from collections import Counter






def draw(X,y,ax,X_samp,title,num,main):

    plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1],
                c='tan', marker='o', s=25, )
    plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1],
                c='darkcyan', marker='o', s=25, )
    X_new = pd.DataFrame(X_samp).iloc[len(X):, :]
    plt.scatter(X_new[0], X_new[1], c='red', s=35, marker='+')
    a = ['(a)','(b)','(c)','(d)','(e)','(f)','(g)','(h)']
    b = ['(i)','(j)','(k)','(l)','(m)','(n)','(o)','(p)']


    if main == 2:
        title = a[num%10-1]+' '+title
    elif main ==1:
        title = b[num%10-1]+' '+title
    plt.title(title)




def main_1(data=1):
    all = [
        'SMOTE_ENN',
        'SMOTE_TomekLinks',
        'SMOTE_RSB',
        'SMOTE_IPF',
    ]
    if data == 2:X, y = load_data(data_name='make_moons')
    elif data == 1:X,y = fourclass_data()
    elif data ==3:X,y = load_data(data_name='make_circles')


    oversamplers_2 = sv2.get_all_oversamplers(all=all)     #加权采样
    oversamplers_1 = sv1.get_all_oversamplers(all=all)     #原始采样
    num=240
    plt.figure(figsize=(20, 10))

    for o in zip(oversamplers_1,oversamplers_2):
        num +=1
        oversampler_1 = o[0]()      #原始采样
        oversampler_2 = o[1]()      #加权采样
        X_samp_1, y_samp_1= oversampler_1.sample(X, y)
        X_samp_2, y_samp_2 = oversampler_2.sample(X, y)


        ax= plt.subplot(num)
        draw(main=1,num=num,X=X,y=y,X_samp=X_samp_1,ax=ax,title=oversampler_1.__class__.__name__)

        num+=1
        ax = plt.subplot(num)
        draw(main=1,num=num,X=X, y=y, X_samp=X_samp_2, ax=ax,title='Weight-'+str(oversampler_2.__class__.__name__))

    # plt.savefig(fname='./pdf/'+'enn_tome_rsb_ipf_smote'+'.pdf',format='pdf',bbox_inches='tight')

    plt.show()


def main_2(data=1):
    all = [
        'SMOTE',
    ]
    

    if data == 2:X, y = load_data(data_name='make_moons')
    elif data == 1:X,y = fourclass_data()
    elif data ==3:X,y = load_data(data_name='make_circles')


    oversamplers_2 = sv2.get_all_oversamplers(all=all)     #加权采样
    oversamplers_1 = sv1.get_all_oversamplers(all=all)     #原始采样
    num=240
    plt.figure(figsize=(20, 10))

    for o in zip(oversamplers_1,oversamplers_2):
        oversampler_1 = o[0]()      #原始采样
        oversampler_2 = o[1]()      #加权采样

        X_samp_1, y_samp_1= oversampler_1.sample(X, y)
        X_samp_2, y_samp_2 = oversampler_2.sample(X, y)

        num += 1
        ax= plt.subplot(num)
        draw(main=2,num=num,X=X,y=y,X_samp=X_samp_1,ax=ax,title=oversampler_1.__class__.__name__)

        num+=1
        ax = plt.subplot(num)
        draw(main=2,num=num,X=X, y=y, X_samp=X_samp_2, ax=ax,title='Weight-'+str(oversampler_2.__class__.__name__))



    '''boderline_1'''
    sm_1 = over_sampling.BorderlineSMOTE(random_state=42, kind="borderline-1")
    X_res, y_res = sm_1.fit_resample(X, y)
    num+=1
    ax = plt.subplot(num)
    draw(main=2,num=num,X=X, y=y, X_samp=X_res, ax=ax, title='borderline_SMOTE1')



    '''weight-boderline'''
    sm_zhou = all_smote_v1.BorderlineSMOTE(random_state=42, kind="weight-borderline-smote", )
    X_res, y_res = sm_zhou.fit_resample(X, y)
    num += 1
    ax = plt.subplot(num)
    draw(main=2,num=num,X=X, y=y, X_samp=X_res, ax=ax, title='Weight-Borderline_SMOTE1')


    all = [
        'ADASYN',
    ]

    oversamplers_2 = sv2.get_all_oversamplers(all=all)     #加权采样
    oversamplers_1 = sv1.get_all_oversamplers(all=all)     #原始采样
    for o in zip(oversamplers_1,oversamplers_2):
        oversampler_1 = o[0]()      #原始采样
        oversampler_2 = o[1]()      #加权采样


        X_samp_1, y_samp_1= oversampler_1.sample(X, y)
        X_samp_2, y_samp_2 = oversampler_2.sample(X, y)
        print(str(oversampler_1.__class__.__name__)+'\tshape %s' % Counter(y_samp_1), '\n\n\n')
        print(str(oversampler_2.__class__.__name__)+'-weight'+'\tshape %s' % Counter(y_samp_2), '\n\n\n')

        num += 1
        ax= plt.subplot(num)
        draw(main=2,num=num,X=X,y=y,X_samp=X_samp_1,ax=ax,title=oversampler_1.__class__.__name__)

        num+=1
        ax = plt.subplot(num)
        draw(main=2,num=num,X=X, y=y, X_samp=X_samp_2, ax=ax,title='Weight-'+str(oversampler_2.__class__.__name__))




    '''kmeans-smote'''
    sm_4 = over_sampling.KMeansSMOTE(random_state=42, )
    X_res, y_res = sm_4.fit_resample(X, y)
    num += 1
    ax = plt.subplot(num)
    draw(main=2,num=num,X=X, y=y, X_samp=X_res, ax=ax, title='kmeans_SMOTE')





    '''weight-kmeans-smote'''
    sm_3 = all_smote_v2.KMeansSMOTE(random_state=42, kind='kmeans-borderline')
    X_res, y_res = sm_3.fit_resample(X, y)
    num+=1
    ax = plt.subplot(num)
    draw(main=2,num=num,X=X, y=y, X_samp=X_res, ax=ax, title='Weight-kmeans_SMOTE')




    # plt.savefig(fname='./pdf/'+'border_km_ADS_smote'+'.pdf',format='pdf',bbox_inches='tight')
    plt.show()




def legend():
    plt.scatter(1,1,label='minority class',
                c='tan', marker='o', s=25, )
    plt.scatter(1,1,label='majority class',
                c='darkcyan', marker='o', s=25, )
    plt.scatter(1,1,label='new samplers',
                c='red', s=35, marker='+')
    axes = plt.axes()
    axes.spines['top'].set_visible(False)
    axes.spines['right'].set_visible(False)
    axes.spines['bottom'].set_visible(False)
    axes.spines['left'].set_visible(False)
    axes.set_xticks([])
    axes.set_yticks([])
    plt.legend(ncol=3,frameon=False)    
    # plt.savefig(fname='./pdf/'+'lengend'+'.pdf',format='pdf',bbox_inches='tight')        
    plt.show()


if __name__ == "__main__":
    main_1(data=1)
    # main_2(data=3)
    # legend()    