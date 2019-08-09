
# coding: utf-8

# In[3]:


import cv2
import numpy as np
import matplotlib.pyplot as plt
#import sklearn.metrics as sk
#import scipy.signal
import os
import sys


# In[2]:


def IrisLocalization(image):
    """give a file path to an iris image, return two lists of parameters of the
    inner and outer circles. The first list is the inner circle and the second
    list is the outer circle. The first number is x coordinate of the circle, 
    the second number is the y coordinate, and the last one is the radius of the
    circle."""
    
    # Step 1: Project the image on x-axis and y-axis, the minima are considered 
    # the center of the pupil.
    
    #plt.imshow(image)
    #plt.show()
    xp = np.argmin(np.sum(image, axis=0)[100:180])+100
    yp = np.argmin(np.sum(image, axis=1)[100:180])+100
    
    
    # Step 2: Find a reasonable threshold to find a more accurate pupil 
    # coordinates. Repeat the step twice for a more accurate estimate.
    
    for i in range(2):
        region = image[yp-60:yp+60, xp-60:xp+60]
        retval, dst = cv2.threshold(region, 65, 1, cv2.THRESH_BINARY)
        mask = np.where(dst != 0, 1, 0)
        xp += np.argmin(np.sum(mask, axis=0)) - 60
        yp += np.argmin(np.sum(mask, axis=1)) - 60
    
    # Step 3: Use Canny edge detection and Hough transform to find two circles.
    
    # set two smaller regions to detect edges faster
    width1 = 70
    region_inner = image[max(0, yp-width1):min(280, yp+width1), 
                       max(0, xp-width1):min(320, xp+width1)]
    width2 = 125
    region_outer = image[max(0, yp-width2):min(280, yp+width2), 
                       max(0, xp-width2):min(320, xp+width2)]
    
    # get two parameters for canny edge detection for the inner circle
    var = 0.33
    median = np.median(region_inner)
    para1 = int(max(0, (1.0 - var) * median))
    para2 = int(min(255, (1.0 + var) * median))
    
    # remove noise caused by eye lashes
    inner_filter = cv2.bilateralFilter(region_inner,9,95,95) 
    # use canny edge detector to get an image of inner boundary
    edged_inner = cv2.Canny(inner_filter, para1, para2)
    inner_circles = cv2.HoughCircles(edged_inner, cv2.HOUGH_GRADIENT, 1,300, 
                                     param1=50, param2=10, minRadius=25, 
                                     maxRadius=58)
    region_outer = cv2.bilateralFilter(region_outer,9,95,95) 
    outer_circles = cv2.HoughCircles(region_outer, cv2.HOUGH_GRADIENT, 1,300, 
                                     param1=30, param2=10, minRadius=95, 
                                     maxRadius=114)
    
    # draw circles
    # pupil boundary
    for i in inner_circles[0,:]:
        inner_circle = [int(i[0])+xp-width1, int(i[1])+yp-width1, i[2]]
    # iris boundary
    for i in outer_circles[0,:]:
        outer_circle = [int(i[0])+xp-width2, int(i[1])+yp-width2, i[2]]
    #plt.imshow(image)
    #plt.show()
    return (inner_circle, outer_circle)

