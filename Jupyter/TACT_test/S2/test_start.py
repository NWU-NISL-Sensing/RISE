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
    
    
    min_max_scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(-1,1))
    myclassifier = [svm.SVC(probability = True, break_ties=True, decision_function_shape='ovr', random_state=0),
                    sklearn.ensemble.RandomForestClassifier(n_estimators=100,random_state=0),
                    DecisionTreeClassifier(random_state=0),neighbors.KNeighborsClassifier(n_neighbors=10),
                    LogisticRegression(random_state=0),GradientBoostingClassifier(n_estimators=100,random_state=0),
                    LinearDiscriminantAnalysis(), AdaBoostClassifier(),
                    GaussianNB(),QuadraticDiscriminantAnalysis()]   
    
    
    times = ['pos1'] ##test set
    train_name = ['pos2','pos3','pos4','pos5'] ##train set
    filepath = r'/root/RISE-Version2/Jupyter/TACT_test/S2/data/' 
    filename = ['_dtw_rssi_12D']
    class_index = 1
    class_num = 6 
    
    
    ##load test data
    #print('\n---------------test data is '  +  times[0] + ' scenario-------------\n')
    data = sio.loadmat(filepath + 'rfid_' + times[0] + filename[0]  + '.mat')
    label = sio.loadmat(filepath + 'label30.mat')
    xx2 = data['dtw_rssi_12D']
    yy2 = label['label30']
    yy2 = yy2.flatten()
    test_x = xx2
    test_y = yy2
    
    ##load train data
    #print('\n-------training data is ' + str(train_name) + ' scenario----------\n')
    xx1 = np.empty(shape=[0, xx2.shape[1]])
    yy1 = np.empty(shape=[1, 0],dtype=int)            
    for ii in train_name:
        data = sio.loadmat(filepath + 'rfid_' + ii + filename[0] + '.mat')
        label = sio.loadmat(filepath + 'label30.mat')
        x1 = data['dtw_rssi_12D']
        y1 = label['label30']
        x1 = min_max_scaler.fit_transform(x1)
        xx1 = np.append(xx1, x1, axis=0)
        yy1 = np.append(yy1, y1, axis=1)
    yy1 = yy1.flatten()
    
    index = [t for t in range(xx1.shape[0])] 
    random.shuffle(index)
    x_train11 = xx1[index]
    x_train1 = x_train11
    y_train1 = yy1[index]
    y_train1 = y_train1 - 1 
       
    
    ############################ Without RISE  ###############################
    print('\n--------  The performance of the underlying model without RISE --------\n')
    x_test1 = min_max_scaler.fit_transform(test_x)
    y_test1 = test_y
    y_test1 = y_test1 - 1
    clf_dif = myclassifier[class_index]
    clf_dif.fit(x_train1,y_train1)
    acc_dif = clf_dif.score(x_test1,y_test1)
    print('The accuracy without RISE: ',acc_dif)
    y_true_dif, y_pred_dif = y_test1,clf_dif.predict(x_test1)
    test_confusion_matrix = confusion_matrix(y_true_dif, y_pred_dif)
    print('Confusion matrix without RISE: \n',test_confusion_matrix)
    
    return x_train1, y_train1, x_test1, y_test1, myclassifier, y_true_dif, y_pred_dif,class_num,class_index