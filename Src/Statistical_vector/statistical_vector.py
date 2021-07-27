"""
The Code contains functions to calcualte the statistical vector.
Cross validation and repeated random sampling can be used to 
increase the diversity of training set and calibration set 
and improve the accuracy of probability vector and statistical vector.
Conformal Prediction: 1. https://pypi.org/project/Orange3-Conformal/
2.https://pypi.org/project/nonconformist/
"""
from nonconformist.base import ClassifierAdapter
from nonconformist.nc import  ClassifierNc, MarginErrFunc
from nonconformist.icp import IcpClassifier
import numpy as np


def train_statistical_vector(data_train, label_train, data_cal, label_cal, classification_model, non_Func = MarginErrFunc(), significance=None):
    """Calculate the statistical vector of train set.

    Usage:
        :param: sample:A dataset that is a 2D numpy array
                data_train, label_train:Training datasets and labels
                data_cal, label_cal:Calibration datasets and labels
                classification_model: A classification model
                non_Func: A noncormity function
        :rtype: train_statistical: The statistical vector of the training set
                train_proba: The probability vector of the training set
    """
    model_classi = ClassifierAdapter(classification_model)
    nc_score = ClassifierNc(model_classi, non_Func)
    icp_model = IcpClassifier(nc_score)
    
    
    icp_model.fit(data_train, label_train)
    icp_model.calibrate(data_cal,label_cal)
    train_statistical = icp_model.predict(data_train,significance)
    train_proba = model_classi.predict(data_train)
    
    return train_statistical, train_proba



def  non_condition_p(validation_c,proba_test):
    """
   Given the nonconformity measurements of test samples and calibration samples, 
   calculate the statistical vector of test samples.
   For label-conditional conformal prediction, validation_c refers to the nonconformity measurement 
   of a specific label in the calibration set.
   Usage:
        :param: validation_c:Nonconformity measurement of calibration samples
                non_test:Nonconformity measurement of test samples
        :rtype: p_test:Statistical vector of test samples
    """

    p_test = np.empty((proba_test.shape)) ##Shape of the statistical vector
    for i in range(proba_test.shape[1]): ##Each column represents a class
        validation_c1 = np.array(validation_c) 
        n_cal = validation_c1.size 
        for j in range(proba_test[:,i].size):##Number of samples of each class
            nc = proba_test[j,i]            
            idx_left = np.searchsorted(validation_c1, nc, 'left',sorter=np.argsort(validation_c1)) ##Note that validation_c1 should be sorted in ascending order
    				
            idx_right = np.searchsorted(validation_c1, nc,'right',sorter=np.argsort(validation_c1))
    				
            n_gt = n_cal - idx_right
    				
            n_eq = idx_right - idx_left + 1
    
            p_test[j, i] = n_gt / (n_cal+1)
    
#            p_test[j, i] += (n_eq+1) * np.random.uniform(0, 1, 1)) / (n_cal + 1) ##random
            p_test[j, i] += (n_eq+1) / (n_cal + 1)   ##no random
            
    return p_test


def test_statistical_vector_param(data_train, label_train,  data_cal, label_cal, data_test,
                             classification_model, non_Func = MarginErrFunc(), significance=None):
    """Calculate the statistical vector of test set.

    Usage:
        :param: sample:A dataset that is a 2D numpy array
                data_train, label_train:Training datasets and labels
                data_cal, label_cal:Calibration datasets and labels
                data_test:Testing datasets
                classification_model: A classification model             
                non_Func: A noncormity function
        :rtype: validation_nonconformity: Nonconformity measurements of calibration samples 
                test_nc_score: Nonconformity measurements of test samples 
                test_proba: The probability vector of the test sample
    """
    model_classi = ClassifierAdapter(classification_model)
    nc_score = ClassifierNc(model_classi, non_Func)
    icp_model = IcpClassifier(nc_score)
    
    
    icp_model.fit(data_train, label_train)
    validation_nonconformity = icp_model.calibrate(data_cal, label_cal)[0][::-1]
    test_proba = model_classi.predict(data_test)
    test_nc_score = icp_model.get_test_nc(data_test)
#    test_statistical = non_condition_p(cal_scores2_,test_nc_score) 
    
    
    return validation_nonconformity, test_nc_score, test_proba



def test_validation_nonconformity(data_train, label_train, data_cal, label_cal, classification_model, non_Func = MarginErrFunc(), significance=None):
    """Calculate the nonconformity measurements of calibration samples.

    Usage:
        :param: sample:A dataset that is a 2D numpy array
                data_train, label_train:Training datasets and labels
                data_cal, label_cal:Calibration datasets and labels
                classification_model: A classification model
                non_Func: A noncormity function
        :rtype: validation_nonconformity: Nonconformity measurements of calibration samples   
    """
    model_classi = ClassifierAdapter(classification_model)
    nc_score = ClassifierNc(model_classi, non_Func)
    icp_model = IcpClassifier(nc_score)
    

    icp_model.fit(data_train, label_train)
    validation_nonconformity = icp_model.calibrate(data_cal, label_cal)[0][::-1]
    
    return validation_nonconformity



"""
#Test Code
from sklearn import svm
import scipy.io as sio
import numpy as np
from sklearn.model_selection import train_test_split

original_sample = sio.loadmat('wig9_label_startend30p1_.mat')
data = original_sample['fe_wig']
label = original_sample['label'].flatten()


data_train, data_cal, label_train, label_cal = train_test_split(data, label, test_size=0.25, random_state=42)
classification_model = svm.SVC(probability = True, break_ties=True, decision_function_shape='ovr', random_state=0)
train_statistical, train_proba = train_statistical_vector(data_train, label_train, data_cal, label_cal, classification_model, non_Func = MarginErrFunc(), significance=None)

"""

