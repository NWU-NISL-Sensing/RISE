"""
The Code contains functions to calcualte the probability vector.

"""

def cal_proba_vector(sample, classification_model_fit):
    """Calculate the probability vector.

    Usage:
        :param: sample:A dataset that is a 2D numpy array
                classification_model_fit: A classification model that has been fitted
        :rtype: 
    """
    proba_vector = classification_model_fit.predict_proba(sample)
    return proba_vector




"""
#Test Code
from sklearn import svm
import scipy.io as sio
original_sample = sio.loadmat('wig9_label_startend30p1_.mat')
data = original_sample['fe_wig']
label = original_sample['label'].flatten()
classification_model = svm.SVC(probability = True, break_ties=True, decision_function_shape='ovr', random_state=0)
classification_model_fit = classification_model.fit(data,label)
proba_vector = cal_proba_vector(data, classification_model_fit)
"""


