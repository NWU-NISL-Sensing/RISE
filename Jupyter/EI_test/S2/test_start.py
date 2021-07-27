# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 22:09:47 2021

@author: Apple
"""
def start(filepath_AR):
    import tensorflow.keras
    import numpy as np
    import sklearn.ensemble
    from sklearn import svm
    from sklearn.metrics import confusion_matrix
    import joblib
    from sklearn import neighbors
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.tree import DecisionTreeClassifier
    import random
    from sklearn.linear_model import LogisticRegression 
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.ensemble import  AdaBoostClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
    
    
    import sys
    sys.path.append(r"/root/RISE-Version2/Jupyter/EI_test/")
    from CNN_Ultrasound_GESTURE_ZSJ import CNN_Classifier, read_wifi_train_data 
    
    import os  
    os.environ["CUDA_VISIBLE_DEVICES"] = '3' #use GPU with ID=3 
    os.environ["TF_CPP_MIN_LOG_LEVEL"]='2' #warning and Error   

    
    from tensorflow.keras import backend as K
    K.clear_session()

    class_index = 0 
    class_num = 6 
    max_size = 1500  
    data_size = 150 
    batch_size = 32 
    epochs = 10
    
    
    myclassifier = [CNN_Classifier(data_size = data_size,batch_size = batch_size,epochs = epochs),svm.SVC(probability = True),
                sklearn.ensemble.RandomForestClassifier(),DecisionTreeClassifier(), 
                neighbors.KNeighborsClassifier(n_neighbors=6), LinearDiscriminantAnalysis(),LogisticRegression(),
                GradientBoostingClassifier(),AdaBoostClassifier(),GaussianNB(),QuadraticDiscriminantAnalysis()] 
    
    times = ['P2phase_fre']##test set
    train_name = ['P1phase_fre','P3phase_fre','P4phase_fre','P5phase_fre'] ##train set
    filepath = r'/root/RISE-Version2/Jupyter/EI_test/data/'
    
    ##load test data
    #print('\n---------------test data is' +  times[0] + '--------------\n')
    train_path = filepath + times[0]
    xx2,yy2 = read_wifi_train_data(train_path,max_size)
    test_x = xx2
    test_y = yy2    

    index_test = [t for t in range(test_x.shape[0])] 
    random.shuffle(index_test)
    x_test1 = test_x[index_test]
    y_test1 = test_y[index_test]
    
    #print('\n--------------trainning data is' + str(train_name) + '--------------\n')
    xx1 = np.empty(shape=[0, xx2.shape[1]])
    yy1 = np.empty((0,1) ,int)
    for ii in train_name:
        if ii != times[0]:
            train_path = filepath + ii
            x1,y1 = read_wifi_train_data(train_path,max_size)
            xx1 = np.append(xx1, x1, axis=0)
            yy1 = np.append(yy1, y1)
    yy1 = yy1.flatten()
    
    index = [t for t in range(xx1.shape[0])] 
    random.shuffle(index)
    x_train1 = xx1[index]
    y_train1 = yy1[index]
#    y_train11 = tensorflow.keras.utils.to_categorical(y_train1, class_num)
    ############################ Without RISE  ###############################
    print('\n--------  The performance of the underlying model without RISE --------\n')

    clf_dif = joblib.load(filepath_AR+'clf_dif.model')
    acc_dif = np.load(filepath_AR+'acc_dif.npy')
    print('The accuracy without RISE: ',acc_dif)
    y_true_dif = np.load(filepath_AR+'acc_dif.npy')
    y_pred_dif = np.load(filepath_AR+'acc_dif.npy')    
    test_confusion_matrix = np.load(filepath_AR+'test_confusion_matrix.npy')
    print('Confusion matrix without RISE: \n',test_confusion_matrix)
    
    return x_train1, y_train1, x_test1, y_test1, myclassifier, y_true_dif, y_pred_dif,class_num,class_index