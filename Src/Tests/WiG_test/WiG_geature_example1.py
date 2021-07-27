# -*- coding: utf-8 -*-
#"""
#@author: Shuangjiao Zhai
#"""


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
import sys
sys.path.insert(0,'../..')
from Statistical_vector.statistical_vector import train_statistical_vector, test_statistical_vector_param, non_condition_p


min_max_scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(-1,1))
myclassifier = [svm.SVC(probability = True, break_ties=True, decision_function_shape='ovr', random_state=0),
                sklearn.ensemble.RandomForestClassifier(n_estimators=100,random_state=0),
                DecisionTreeClassifier(random_state=0),neighbors.KNeighborsClassifier(n_neighbors=10),
                LogisticRegression(random_state=0),GradientBoostingClassifier(n_estimators=100,random_state=0),
                LinearDiscriminantAnalysis(), AdaBoostClassifier(),
                GaussianNB(),QuadraticDiscriminantAnalysis()]   


times = ['p1_'] ##test set
train_name = ['p2_','p3_','m1_','m2_'] ##train set
filepath = r'./data/' 
filename = ['wig9_label_startend30']
class_index = 0 
class_num = 6   


##load test data
print('\n---------------test data is '  +  times[0] + ' scenario-------------\n')
data = sio.loadmat(filepath + filename[0] + times[0] + '.mat')
xx2 = data['fe_wig']
yy2 = data['label']
yy2 = yy2.flatten()
test_x = xx2
test_y = yy2

##load train data
print('\n-------training data is ' + str(train_name) + ' scenario----------\n')
xx1 = np.empty(shape=[0, xx2.shape[1]])
yy1 = np.empty(shape=[1, 0],dtype=int)            
for ii in train_name:
    data = sio.loadmat(filepath + filename[0] + ii+ '.mat')
    x1 = data['fe_wig']
    y1 = data['label']
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


############################ Trainning Anomaly Detector  ###############################  

## Probability vector and statistical vector are calculated according to the training data
skf = StratifiedKFold(n_splits=3, random_state=0,shuffle=True)
cal_proba = np.empty(shape=[0, (class_num)*4])
cal_score = np.empty(shape=[0, 1*4])
train_proba = np.empty(shape=[0, (class_num)*4])
train_nonconformity = np.empty(shape=[0, (class_num)*4])
train_p = np.empty(shape=[0, (class_num)*4])
cal_label = np.empty(shape=[0,1])
train_label = np.empty(shape=[0,1])
for train_index, cal_index in skf.split(x_train1, y_train1):
    data_train = x_train1[train_index, :] 
    label_train = y_train1[train_index]
    data_cal = x_train1[cal_index, :]
    label_cal = y_train1[cal_index]
    
    train_p1_svm_, train_proba1_svm_ = train_statistical_vector(data_train, label_train, data_cal, label_cal, 
                       classification_model = myclassifier[0], non_Func = MarginErrFunc(), significance=None) 
    train_p1_rf_, train_proba1_rf_ = train_statistical_vector(data_train, label_train, data_cal, label_cal, 
                       classification_model = myclassifier[1], non_Func = MarginErrFunc(), significance=None) 
    train_p1_lr_, train_proba1_lr_ = train_statistical_vector(data_train, label_train, data_cal, label_cal, 
                       classification_model = myclassifier[4], non_Func = MarginErrFunc(), significance=None)
    train_p1_gbc_, train_proba1_gbc_ = train_statistical_vector(data_train, label_train, data_cal, label_cal, 
                       classification_model = myclassifier[5], non_Func = MarginErrFunc(), significance=None)
    
    train_proba1 = np.hstack((train_proba1_svm_, train_proba1_rf_, train_proba1_lr_, train_proba1_gbc_))
    train_p1 = np.hstack((train_p1_svm_, train_p1_rf_, train_p1_lr_, train_p1_gbc_))
    train_proba = np.append(train_proba, train_proba1, axis=0)  
    train_p = np.append(train_p, train_p1, axis=0)  
    train_label = np.append(train_label,y_train1[train_index]) 
    
p_thr = 0.1
rows_ = []  
for row_ in range(train_p.shape[0]):
    if (max(train_p[row_,0:0+class_num]) > p_thr) & (max(train_p[row_,class_num:class_num*1+class_num]) > p_thr) \
        & (max(train_p[row_,class_num*2:class_num*2+class_num]) > p_thr) & (max(train_p[row_,class_num*3:class_num*3+class_num]) > p_thr) :               
            rows_.append(row_)
        
pro_n_p = np.hstack((train_proba, train_p))
group_it = pro_n_p[rows_,:] 
train_label1 = train_label[rows_] 

group_it = preprocessing.scale(group_it)
   
index = [t for t in range(group_it.shape[0])] 
random.shuffle(index)
group_it1 = group_it[index] 
train_p_lable1 = train_label1[index]



####Training an anomaly detector for each class

##Generalize probability vectors and statistical vectors for each class
names = locals()
fe_p_prob = np.hstack((train_p_lable1.reshape(-1,1),group_it1))
for tt in range(class_num):
    names['fe_p_prob_%s'%tt] = fe_p_prob[fe_p_prob[:,0]==tt]
    names['fe_p_prob_%s'%tt] = np.delete(names['fe_p_prob_%s'%tt], 0, 1)  

##training one-class SVM model for each class
for tt in range(class_num):
    names['./save_model/clf_p_prob_%s'%tt] = svm.OneClassSVM(nu=0.5,kernel="linear" )
    names['./save_model/clf_p_prob_%s'%tt].fit(names['fe_p_prob_%s'%tt])
    ##save one-class SVM model for each class
    joblib.dump(names['./save_model/clf_p_prob_%s'%tt],'./save_model/clf_p_prob_%s'%tt+'.model')

##Ensemble Learning
clf_dif1 = sklearn.ensemble.RandomForestClassifier(n_estimators=100,random_state=0)
clf_dif2 = svm.SVC(probability = True, random_state=0)
clf_dif3 = neighbors.KNeighborsClassifier(n_neighbors=10)
clf_dif4 = LogisticRegression(random_state=0)
clf_dif7 = AdaBoostClassifier(random_state=0)   
clf_pit = VotingClassifier(estimators = [('rf',clf_dif1),('svm',clf_dif2),
('knn',clf_dif3),('lr',clf_dif4),('AdaBoost',clf_dif7)],voting='hard')      
clf_pit.fit(group_it1,train_p_lable1)
joblib.dump(clf_pit,'./save_model/clf_pit.model')  



#####Calculate the probability vector and statistical vector of the test set
           
calibration_portion = 0.5
split = StratifiedShuffleSplit(n_splits=1,
                   test_size=calibration_portion)
for train, cal in split.split(x_train1,y_train1):
    cal_scores1_svm = np.empty(cal.reshape(-1,1).shape,dtype=float)
    cal_scores1_rf = np.empty(cal.reshape(-1,1).shape,dtype=float)
    cal_scores1_lr = np.empty(cal.reshape(-1,1).shape,dtype=float)
    cal_scores1_gbc = np.empty(cal.reshape(-1,1).shape,dtype=float)
    test_svmnc1_score = np.empty(np.array([x_test1.shape[0],class_num]),dtype=float)
    test_rfnc1_score = np.empty(np.array([x_test1.shape[0],class_num]),dtype=float)
    test_lrnc1_score = np.empty(np.array([x_test1.shape[0],class_num]),dtype=float)
    test_gbcnc1_score = np.empty(np.array([x_test1.shape[0],class_num]),dtype=float)
    test_svm1_proba = np.empty(np.array([x_test1.shape[0],class_num]),dtype=float)
    test_rf1_proba = np.empty(np.array([x_test1.shape[0],class_num]),dtype=float)
    test_lr1_proba = np.empty(np.array([x_test1.shape[0],class_num]),dtype=float)
    test_gbc1_proba = np.empty(np.array([x_test1.shape[0],class_num]),dtype=float)
    for repeat in range(10):
        train_sample = np.random.choice(train.size, train.size, replace=True)
        data_train = x_train1[train_sample, :]
        label_train = y_train1[train_sample]
        data_cal = x_train1[cal, :]
        label_cal = y_train1[cal]
        data_test = x_test1

        cal_scores_svm, test_svmnc_score, test_svm_proba = test_statistical_vector_param(data_train, label_train,  data_cal, label_cal, data_test,
                 classification_model=myclassifier[0], non_Func = MarginErrFunc(), significance=None)
        cal_scores1_svm = np.hstack((cal_scores1_svm,cal_scores_svm.reshape(-1,1)))
        test_svmnc1_score = np.dstack((test_svmnc1_score,test_svmnc_score))
        test_svm1_proba = np.dstack((test_svm1_proba,test_svm_proba))
  
        cal_scores_rf, test_rfnc_score, test_rf_proba = test_statistical_vector_param(data_train, label_train,  data_cal, label_cal, data_test,
                 classification_model=myclassifier[1], non_Func = MarginErrFunc(), significance=None)
        cal_scores1_rf = np.hstack((cal_scores1_rf,cal_scores_rf.reshape(-1,1)))
        test_rfnc1_score = np.dstack((test_rfnc1_score,test_rfnc_score))
        test_rf1_proba = np.dstack((test_rf1_proba,test_rf_proba))
          
        cal_scores_lr, test_lrnc_score, test_lr_proba = test_statistical_vector_param(data_train, label_train,  data_cal, label_cal, data_test,
                 classification_model=myclassifier[4], non_Func = MarginErrFunc(), significance=None)
        cal_scores1_lr = np.hstack((cal_scores1_lr,cal_scores_lr.reshape(-1,1)))
        test_lrnc1_score = np.dstack((test_lrnc1_score,test_lrnc_score))
        test_lr1_proba = np.dstack((test_lr1_proba,test_lr_proba))
        
        cal_scores_gbc, test_gbcnc_score, test_gbc_proba = test_statistical_vector_param(data_train, label_train,  data_cal, label_cal, data_test,
                 classification_model=myclassifier[5], non_Func = MarginErrFunc(), significance=None)
        cal_scores1_gbc = np.hstack((cal_scores1_gbc,cal_scores_gbc.reshape(-1,1)))
        test_gbcnc1_score = np.dstack((test_gbcnc1_score,test_gbcnc_score))
        test_gbc1_proba = np.dstack((test_gbc1_proba,test_gbc_proba))
        
        
    ##Nonconformity score of the validation set of the test set    
    cal_scores2_svm = np.mean(np.delete(cal_scores1_svm,0,1), axis=1)
    cal_scores2_rf = np.mean(np.delete(cal_scores1_rf,0,1), axis=1)
    cal_scores2_lr = np.mean(np.delete(cal_scores1_lr,0,1), axis=1)
    cal_scores2_gbc = np.mean(np.delete(cal_scores1_gbc,0,1), axis=1) 
    
    ##Nonconformity score of the test set
    test_svmnc2_score = np.mean(np.delete(test_svmnc1_score,0,2), axis=2)
    test_rfnc2_score = np.mean(np.delete(test_rfnc1_score,0,2), axis=2)
    test_lrnc2_score = np.mean(np.delete(test_lrnc1_score,0,2), axis=2)
    test_gbcnc2_score = np.mean(np.delete(test_gbcnc1_score,0,2), axis=2)
    
    ##The probability vector of the test set
    test_proba_svm_ = np.mean(np.delete(test_svm1_proba,0,2), axis=2)
    test_proba_rf_ = np.mean(np.delete(test_rf1_proba,0,2), axis=2)
    test_proba_lr_ = np.mean(np.delete(test_lr1_proba,0,2), axis=2)
    test_proba_gbc_ = np.mean(np.delete(test_gbc1_proba,0,2), axis=2)

##The statistical vector of the test set    
test_p_svm_ = non_condition_p(cal_scores2_svm, test_svmnc2_score)
test_p_rf_ = non_condition_p(cal_scores2_rf, test_rfnc2_score)
test_p_lr_ = non_condition_p(cal_scores2_lr, test_lrnc2_score)   
test_p_gbc_ = non_condition_p(cal_scores2_gbc, test_gbcnc2_score)    
test_group_it = np.hstack((test_proba_svm_, test_proba_rf_, test_proba_lr_, test_proba_gbc_, 
                           test_p_svm_, test_p_rf_, test_p_lr_, test_p_gbc_))
test_group_it = preprocessing.scale(test_group_it) 


##Determine whether the prediction of the original model is correct
discarded_sample = [] 
accept_sample = [] 
discarded_right_sample = [] 
accept_right_sample = [] 
accept_or_reject = []   
dis_test = np.empty((test_group_it.shape[0],class_num),float)
for aa in range(class_num):
    aa_prob = joblib.load('./save_model/clf_p_prob_%s'%aa+'.model')
    dis_test[:,aa] = aa_prob.decision_function(test_group_it).flatten()            
for t in range(len(y_pred_dif)):
        result_it_d = np.argmax(dis_test[t,:])
        svmit = joblib.load('./save_model/clf_pit.model')
        result_it = svmit.predict(test_group_it[t].reshape(1,-1))
        if (y_pred_dif[t] == result_it == result_it_d):
            accept_sample.append(t)
            accept_or_reject.append(1)
            if(y_pred_dif[t] == y_true_dif[t]):
                accept_right_sample.append(t)
        else:
            discarded_sample.append(t)
            accept_or_reject.append(-1)
            if(y_pred_dif[t] == y_true_dif[t]):
                discarded_right_sample.append(t)   
  

reject_num = len(discarded_sample) 
reject_num_right =  len(discarded_right_sample) 
accept_num = len(accept_sample) 
accept_num_right = len(accept_right_sample) 

TP = reject_num - reject_num_right
FP = reject_num_right
FN = accept_num - accept_num_right
TN = accept_num_right

Accuracy = (TP+TN)/(TP+FP+FN+TN)
Precision = TP/(TP+FP)
Recall = TP/(TP+FN)
F1_Score = 2 * Precision * Recall / (Precision + Recall)

print('\n ------------------ The performance of the RISE --------------------\n')
print('True Positive:',TP)
print('False Positive:',FP)
print('False Negative:',FN)
print('True Negative:', TN)
print('Accuracy:',Accuracy)
print('Precision:',Precision)
print('Recall:',Recall)
print('F1_Score:', F1_Score)




print('\n--------  The performance of the underlying model after RISE --------\n')     
if len(accept_sample) == 0:
    print('null')
else:
    accept_sample_index = np.array(accept_sample)
    accept_data = x_test1[accept_sample_index]
    accept_label = y_test1[accept_sample_index]
    clf_dif = myclassifier[class_index]
    clf_dif.fit(x_train1,y_train1)
    acc_aft = clf_dif.score(accept_data,accept_label)
    print('The accuracy with RISE: ',acc_aft)
    y_true_actrain, y_pred_actrain = accept_label,clf_dif.predict(accept_data)
    aft_confusion_matrix = confusion_matrix(y_true_actrain, y_pred_actrain)
    print('Confusion matrix after RISE: \n',aft_confusion_matrix)




