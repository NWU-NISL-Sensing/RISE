# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 22:09:47 2021

@author: Apple
"""
def start():
    import numpy as np
    import scipy.io as sio
    import sklearn.ensemble
    from sklearn import svm
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import confusion_matrix
    from sklearn import preprocessing
    import joblib
    from sklearn import neighbors
    from sklearn.model_selection import StratifiedShuffleSplit
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.tree import DecisionTreeClassifier
    import random
    from sklearn.linear_model import LogisticRegression 
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.ensemble import  AdaBoostClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
    from sklearn.ensemble import VotingClassifier
    from nonconformist.nc import MarginErrFunc
    import warnings
    warnings.filterwarnings("ignore", message="Numerical issues were encountered ")
    import sys
    sys.path.insert(0,'/root/RISE-Version2/')
    from Statistical_vector.statistical_vector import train_statistical_vector, test_statistical_vector_param, non_condition_p
    import sys
    sys.path.append(r"/root/RISE-Version2/Jupyter/M_Touch/data/")
    from feature_ex import touch 
    
    min_max_scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(-1,1))
    myclassifier = [svm.SVC(probability = True, break_ties=True, decision_function_shape='ovr', random_state=0),
                sklearn.ensemble.RandomForestClassifier(n_estimators=100,random_state=0),
                DecisionTreeClassifier(random_state=0),neighbors.KNeighborsClassifier(n_neighbors=5, weights='uniform',
                                     algorithm='auto', leaf_size=30,
                                     p=2, metric='minkowski',
                                     metric_params=None, n_jobs=1),
                LogisticRegression(random_state=0),GradientBoostingClassifier(n_estimators=100,random_state=0),
                LinearDiscriminantAnalysis(), AdaBoostClassifier(),
                GaussianNB(),QuadraticDiscriminantAnalysis()]   
    
    
    nameb=['fh1sz1.txt','fh1zz1.txt','fh1wmz1.txt','fh1xmz1.txt']
    namec=['hy1sz1.txt','hy1zz1.txt','hy1wmz1.txt','hy1xmz1.txt']
    namef=['lp1sz1.txt','lp1zz1.txt','lp1wmz1.txt','lp1xmz1.txt']
    namej=['ty1sz1.txt','ty1zz1.txt','ty1wmz1.txt','ty1xmz1.txt']
    namek=['wht1sz1.txt','wht1zz1.txt','wht1wmz1.txt','wht1xmz1.txt']
    namel=['xyk1sz1.txt','xyk1zz1.txt','xyk1wmz1.txt','xyk1xmz1.txt']
    namem=['yhy1sz1.txt','yhy1zz1.txt','yhy1wmz1.txt','yhy1xmz1.txt']
    nameo=['zyx1sz1.txt','zyx1zz1.txt','zyx1wmz1.txt','zyx1xmz1.txt']
    nameq=['ftc1sz1.txt','ftc1zz1.txt','ftc1wmz1.txt','ftc1xmz1.txt']
    namet=['zmh1sz1.txt','zmh1zz1.txt','zmh1wmz1.txt','zmh1xmz1.txt']
    name=[nameb,namec,namef,namej,namek,namel,namem,nameo,namet]

    times = ['3/'] ##test set
    train_name = ['1/','5/','6/'] ##train set
    filepath = r'/root/RISE-Version2/Jupyter/M_Touch/data/' 
    class_index = 3
    class_num = 9  
    
    
    ##load test data
    #print('\n---------------test data is '  +  times[0] + ' scenario-------------\n')
    xx2,yy2 = touch(filepath+times[0],name)
    yy2  = np.array(yy2).flatten()
    test_x = xx2
    test_y = yy2
    
    ##load train data
    #print('\n-------training data is ' + str(train_name) + ' scenario----------\n')
    xx1 = np.empty(shape=[0, xx2.shape[1]])
    yy1 = np.empty(shape=[1, 0],dtype=int) 
    yy1 = yy1.flatten()           
    for ii in train_name:
        x1,y1=touch(filepath+ii,name)
        y1 = np.array(y1).flatten()
        x1 = min_max_scaler.fit_transform(x1)
        xx1 = np.append(xx1, x1, axis=0)
        yy1 = np.append(yy1, y1, axis=0)
    yy1 = yy1.flatten()
    
    index = [t for t in range(xx1.shape[0])] 
    random.shuffle(index)
    x_train11 = xx1[index]
    x_train1 = x_train11
    y_train1 = yy1[index]
    #y_train1 = y_train1 - 1 
       
    
    ############################ Without RISE  ###############################
    print('\n--------  The performance of the underlying model without RISE --------\n')
    x_test1 = min_max_scaler.fit_transform(test_x)
    y_test1 = test_y
    #y_test1 = y_test1 - 1
    clf_dif = myclassifier[class_index]
    clf_dif.fit(x_train1,y_train1)
    acc_dif = clf_dif.score(x_test1,y_test1)
    print('The accuracy without RISE: ',acc_dif)
    y_true_dif, y_pred_dif = y_test1,clf_dif.predict(x_test1)
    test_confusion_matrix = confusion_matrix(y_true_dif, y_pred_dif)
    print('Confusion matrix without RISE: \n',test_confusion_matrix)
    
    return x_train1, y_train1, x_test1, y_test1, myclassifier, y_true_dif, y_pred_dif,class_num,class_index