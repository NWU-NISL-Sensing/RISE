# -*- coding: utf-8 -*-
"""
Created on Sat Sep  7 21:25:44 2019

@author: jiao
"""


import scipy.io as sio
import os
import numpy as np
import pandas as pd
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from keras.layers import Dense,Dropout,Flatten,Conv1D,MaxPooling1D,Activation,Reshape,BatchNormalization
import matplotlib.pyplot as plt
from keras.wrappers.scikit_learn import KerasClassifier



class_num = 6  ##number of classes
data_size = 150 ##length of the input layer
batch_size = 32
epochs = 15


'''
different actions
'''
string_action_1='gesture1_'
string_action_2='gesture2_'
string_action_3='gesture3_'
string_action_4='gesture4_'
string_action_5='gesture5_'
string_action_6='gesture6_'


'''
wifi_file_to_label() Identify the six actions by filename to get the corresponding label
file_path：e.g., lm_p1_gesture4_6.dat.mat, the label corresponding to gesture4_ is this label
'''
def wifi_file_to_label(file_path):
    if(file_path.find(string_action_1) != -1):
        return 0
    elif(file_path.find(string_action_2) != -1):
        return 1
    elif(file_path.find(string_action_3) != -1):
        return 2
    elif(file_path.find(string_action_4) != -1):
        return 3
    elif(file_path.find(string_action_5) != -1):
        return 4
    elif (file_path.find(string_action_6) != -1):
        return 5



'''
standardized
'''

def feature_normalize(dataset):
    
    return dataset
#def feature_normalize(dataset):
#    mu = np.mean(dataset, axis=0)
#    sigma = np.std(dataset, axis=0)
#    return (dataset - mu) / sigma


'''
read all the data under the folder dir_path and generate training data and labels
'''
def read_wifi_test_data(dir_path,max_size):
    result=[]
    for main_test_dir, test_subdir, test_file_name_list in os.walk(dir_path):
        for test_filename in test_file_name_list:
            test_path = os.path.join(main_test_dir, test_filename)
            if(test_filename.find('.mat') != -1):
                result.append(test_path)
#    print(result)
    
    label_tmp = -1
    flag = 0
    labels = []
    under_simple = []
    data_array = np.zeros((1, max_size*1))
    for file_tmp in result:
        label_tmp = wifi_file_to_label(file_tmp) 
        data_file = sio.loadmat(file_tmp) 
        data_tmp = data_file['phase_fre_data']
        if(len(data_tmp)<max_size):
            num=max_size-len(data_tmp)
            arrany_zero=np.zeros((num, 1))
            data_tmp=np.vstack((data_tmp,arrany_zero))
        if(len(data_tmp)>max_size):
            data_tmp=data_tmp[:max_size,:]
            
        

###subsample, take 1 out of 10 samples
        for i in range(0, max_size, 10):
            under_simple.append(data_tmp[i])
        data_tmp = np.array(under_simple)
        under_simple=[]


        data_tmp_frame = pd.DataFrame(data_tmp)
        for i in range(1):
            data_tmp_frame[i] = feature_normalize(data_tmp_frame[i])
        data_tmp = data_tmp_frame.values
#        print(data_tmp.shape)
        data_tmp = data_tmp.flatten()
#        print(data_tmp.shape)
        if (flag == 0):
            data_array = data_tmp
            flag = 1
        else:
            data_array = np.vstack((data_array,data_tmp))
        labels.append(label_tmp)
    labels = np.array(labels)
    test_x = data_array
    return test_x,labels
    



'''
get training samples and labels
'''
def read_wifi_train_data(dir_path,max_size):
    return read_wifi_test_data(dir_path,max_size)






class CNN_Classifier(KerasClassifier):
    def __init__(self,datasize = 300,**sk_params):
       # KerasClassifier.__init__(self)
        #super(CNN_Classifier, self).__init__()
        self.data_size=datasize
        self.build_fn=self.create_model
        self.sk_params = sk_params

    def create_model(self):
        model = Sequential()
        model.add(Reshape((data_size, 1), input_shape=(data_size*1,)))  
        model.add(Conv1D(100, 10,  input_shape=(data_size, 1,)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling1D(3))
        model.add(Conv1D(100, 10))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling1D(3))
        model.add(Conv1D(160, 10))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling1D(3))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(128, activation='tanh'))
        model.add(Dense(32, activation='tanh'))
        model.add(Dense(class_num, activation='softmax'))
        print(model.summary())
        model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
        #model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
        return model

