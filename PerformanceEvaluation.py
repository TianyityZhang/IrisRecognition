
# coding: utf-8

# In[ ]:


import cv2
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as sk
import scipy.signal
import os
import sys
from IrisLocalization  import *
from IrisNormalization import *
from ImageEnhancement import *
from FeatureExtraction import *
from IrisMatching      import *


# In[ ]:


x = np.arange(1,11)*10

# Get the performance for different feature vector dimentions as well as different distance 
# measure methods
l1norm = []
l2norm = []
CosSimilarity = []
for i in range(10, 101, 10):
    l1norm.append(IrisMatching(trainfeature, testfeature, 3, n_components = i))
    l2norm.append(IrisMatching(trainfeature, testfeature, 2, n_components = i))
    CosSimilarity.append(IrisMatching(trainfeature, testfeature, 1, n_components = i))
    
# The starting point for dimensions are 10, the end point is 100
group_labels = ["10","20","30","40","50","60","70","80","90","100"]
plt.title('ROC curve of 3 distance measurement')
plt.xlabel('n_components')
plt.ylabel('accuracy')
 

plt.plot(x, l1norm,'r', label='l1norm')
plt.plot(x, l2norm,'b',label='l2norm')
plt.plot(x, CosSimilarity ,'g',label='CosSimilarity')

plt.xticks(x, group_labels, rotation=0)
 
plt.legend(bbox_to_anchor=[0.3, 1])
plt.grid()
plt.show()

