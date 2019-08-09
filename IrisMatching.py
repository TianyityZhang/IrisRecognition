
# coding: utf-8

# In[ ]:


import cv2
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as sk
import scipy.signal
import os
import sys


# In[1]:


def IrisMatching(trainfeature, testfeature, distanceMeasure=1, n_components = 80):
    """for distance measure, 1 is cosine similarity, 2 is euclidean distance, 3 is manhattan distance.
    give train feature and test feature, specify distance measure method, and the number of components in LDA, 
    return the prediction accuracy."""
    
    # response is the y, i.e. the true number of eyes for each feature vector
    response = np.repeat(np.arange(1,109), 21)
    
    # use the feature vector and response to do linear discriminant analysis.
    clf = LinearDiscriminantAnalysis(n_components=n_components)
    clf.fit(trainfeature, response)
    
    # reduce the dimension to the specified number
    new_train_feat = clf.transform(trainfeature)
    new_test_feat = clf.transform(testfeature)
    
    result = []
    if distanceMeasure == 1:
        for i in range(new_test_feat.shape[0]):
            result.append(response[np.argmax(sk.pairwise.cosine_similarity(new_train_feat,new_test_feat[i,].reshape(1,n_components)))])
    elif distanceMeasure == 2:
        for i in range(new_test_feat.shape[0]):
            result.append(response[np.argmin(sk.pairwise.euclidean_distances(new_train_feat,new_test_feat[i,].reshape(1,n_components)))])
    elif distanceMeasure == 3:
        for i in range(new_test_feat.shape[0]):
            result.append(response[np.argmin(sk.pairwise.manhattan_distances(new_train_feat,new_test_feat[i,].reshape(1,n_components)))])
    
    
    # compare answers to get accuracy
    accuracy = sum(np.array(result) == np.arange(new_test_feat.shape[0])+1)/new_test_feat.shape[0]
    return accuracy

