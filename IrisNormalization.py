
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


def IrisNormalization(circle_p, circle_i,image,theta = 0):
    """give the return value of IrisLocalization, return the normalized image, 
    which is 64*512 """
    
    mapx = np.ndarray((64,512),dtype=np.float32)
    mapy = np.ndarray((64,512),dtype=np.float32)
    for i in range(64):
        for j in range(512):
            theta1 = (2*np.pi*j/1024)+theta-np.pi/2
            ratio = i/128
            xp = circle_p[0]+circle_p[2]*np.sin(theta1)
            yp = circle_p[1]+circle_p[2]*np.cos(theta1)
            xi = circle_i[0]+circle_i[2]*np.sin(theta1)
            yi = circle_i[1]+circle_i[2]*np.cos(theta1)
            mapx[i,j] = xp+ratio*(xi-xp)
            mapy[i,j] = yp+ratio*(yi-yp)
    result = cv2.remap(image,mapx,mapy,cv2.INTER_LINEAR)
    return result

