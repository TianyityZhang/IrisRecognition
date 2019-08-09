
# coding: utf-8

# In[ ]:


import cv2
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as sk
import scipy.signal
import os
import sys


# In[ ]:


def ImageEnhancement(image):
    """give the normalized image, return an enhanced image"""
    
    # Step1: Take average of 16*16 window as approx. illumination
    #illum = np.ndarray((4,32),dtype = np.float32)
    #for i in range(4):
    #    for j in range(32):
    #        illum[i,j] = np.sum(image[(16*i):(16*(i+1)),(16*j):(16*(j+1))])/256
            
    # Step2: Expand the illumination size to 64*512 by bicubic interpolation.
    #illum_whole = cv2.resize(illum, None,fx=16, fy=16, interpolation=cv2.INTER_CUBIC)
    
    # Step3: Subtract the illumination to compensate for lighting conditions.
    #for i in range(64):
    #    for j in range(512):
    #        image[i, j] -= illum_whole[i, j]
            
    # Step4: Enhance the lighting corrected image by histogram equalization
    
    # Normal histogram equalization
    #equalized_image = cv2.equalizeHist(image)
    for i in range(2):
        for j in range(16):
            image[(32*i):(32*(i+1)),(32*j):(32*(j+1))] = cv2.equalizeHist(image[(32*i):(32*(i+1)),(32*j):(32*(j+1))])
    # Adaptive histogram equalization
    #clahe = cv2.createCLAHE(clipLimit=3, tileGridSize=(32,32))
    #equalized_image = clahe.apply(image)
    
    return image

