
# coding: utf-8

# In[1]:


import cv2
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as sk
import scipy.signal
import os
import sys


# In[2]:


def modulating_function(x, y, f):
    m = np.cos(2*np.pi*f*np.sqrt(x**2+y**2))
    return m


# In[3]:


def kernel(x, y, f, sigmaX, sigmaY):
    g = (1/(2*np.pi*sigmaX*sigmaY))*np.exp(-0.5*(x**2/sigmaX**2 + 
                                                 y**2/sigmaY**2))*\
    modulating_function(x, y, f)
    return g


# In[4]:


def spatial_filter(f, sigmaX, sigmaY):
    s_filter = np.zeros((8, 8))
    for i in range(8):
        for j in range(8):
            s_filter[i, j] = kernel((j), (i), f, sigmaX, sigmaY)
    return s_filter


# In[5]:


def FeatureExtraction(image):
    """give the enhanced image, return the feature vector of the image"""
    
    # Step 1: Find the region of interest.
    roi = image[:48,:]
    
    # Step 2: Get two filtered image.
    filter1 = spatial_filter(0.6, 3, 1.5)
    filtered1 = scipy.signal.convolve2d(roi, filter1, mode='same')
    filter2 = spatial_filter(0.6, 4.5, 1.5)
    filtered2 = scipy.signal.convolve2d(roi, filter2, mode='same')
    
    #Step 3: Get the feature vector.
    feature_vec = np.zeros(1536)
    for i in range(2):
        filtered = [filtered1, filtered2][i]
        for j in range(6):
            for k in range(64):
                mean = np.mean(abs(filtered[j*8:(j+1)*8, k*8:(k+1)*8]))
                sd = np.mean(abs(abs(filtered[j*8:(j+1)*8, k*8:(k+1)*8])- mean))
                feature_vec[i*768 + 128*j + 2*k] = mean
                feature_vec[i*768 + 128*j + 2*k + 1] = sd        
    return feature_vec.reshape(1,-1)

