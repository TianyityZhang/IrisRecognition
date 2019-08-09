Iris Localization:
We followed what described in the paper, except for several modifications. 
Step 1: Project the image on x-axis and y-axis, the minima are considered  the center of the pupil.
Step 2: Find a reasonable threshold to find a more accurate pupil coordinates. Repeat the step twice for a more accurate estimate.
Step 3: Use Canny edge detection and Hough transform to find two circles.
In step 1, we didn’t calculate the sum along all the columns and rows for the whole image. Some people have thick eyelashes that can affect the preliminary center of the pupil. We get a smaller subimage to minimize the problems caused by eyelashes.
In step 2, we use a threshold to get a mask to filter out the inner circle, which is the inner boundary of the iris. To minimize the effect of eyelashes, we apply a bilateral filter to blur the image.


Iris Normalization:
This function can transform the iris to a rectangle. As described in the iris matching part of the paper, we want to obtain ‘approximate rotation invariance’ by ‘unwrapping the iris ring at different initial angles’. The initial angle is an input for the normalization function. 


Image Enhancement:
We implement two different methods. The first one is as described in the paper. First reduce the background illumination, then do adaptive histogram normalization. However, this doesn’t work well, so we use the normal histogram equalization function on each small segmentation of the normalized image. This can also compensate for the unequal background illumination. 
                                
Feature Extraction:
For feature extraction, we follow what’s described in the paper. 
Step 1: Find the region of interest, according to the paper, it should be the upper 48 rows of the original input image.
Step 2: Get two filtered image. We use the defined modulating function instead of the Gabor modulating function. We also         use the given parameters for sigmaX and sigmaY.
However, f value is unspecified. We tune the parameter on our own. 


Iris Matching:
Generally we used Fisher LDA method to reduce dimension and clustered the input test data by nearest center clustering.
Step 1: Build full input training feature (108*3 photos*7 rotation angle) and response.
Step 2: Select input testing feature, which is the best quality image in folder 2 with respect to each person.
Step 3: Implement LDA to training data and use the model to reduce dimension of test data.
Step 4: Use array manipulation to find the optimal cluster of test data efficiently.


Performance Evaluation:
We plot a line chart showing the trending of accuracy of our algorithm. As n_components goes up, the accuracy becomes better at first and then level-off.


Iris Recognition:
In this main file, we assemble everything. 
Step 1: read images and get the best quality images
First, we use a helper function to read all the images, then we get the list of all the best images in both the training set and the testing set. Since with one best quality image for each eye, we cannot achieve a good accuracy, we do not use the best image in the training set alone to get all the feature vectors. However, to get a better accuracy, we do use the best quality image in the testing set to test. 
Step 2: for each image, localize the iris with IrisLocalization, normalize it with IrisNormalization, enhance it with IrisEnhancement.
Step 3: use a helper function to train all the feature vectors, the use iris matching to get the accuracy rate.
Step 4: plot the accuracy rate as an ROC curve.
        
                
Limitations:
1. We try to do image selection, however, due to the small sample or due to the selection process, after we select the best quality images, the accuracy rate is only 40%. 
2. We try not to use fisher discriminant but use all the features to do selection, however, all the features not only run very slow but also return a very low accuracy rate.
3. The outer boundary of the iris is a lighter edge compare to the eyelashes. Hough transformation tends to find circles around the eyelashes.We want to preprocess the image, reduce the effect of the eyelashes, enhance the outer boundary, then do edge detection. Now we use bilateral filter, however it’s not very effective.