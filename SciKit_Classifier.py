# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 15:28:09 2020
Train a linear SVC (or other SciKit classifiers) for car detection in training
images separated in the corresponding classes
0: no car
1: car

1. color_hist()
Function that creates histogram of each color in the picture

2. bin_spatial()
Downsize image and shape it to a simple feature vector

3. hog_features()
Detects gradients in image and summarizes them to feature vector

4. extract_features()
Function to call to extract and combine desired features from all images.
Initialize Settings for Feature extraction here

train_classifier()
Main function calling all of the above functions to extract selected Features,
train and test the linear SVC.
Features can be added/omitted individually.
@author: Thomas
"""

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
from skimage.feature import hog
import pickle

# Compute Color Histogram and return as feature vector
def color_hist(image, nbins, bins_range, cspace):
    # apply color conversion if other than 'RGB'
    if cspace!='RGB':
        string = 'img = cv2.cvtColor(img, cv2.COLOR_RGB2'+cspace+')'
        eval(string)
    # Compute the histogram of the RGB channels separately
    rhist = np.histogram(image[:,:,0], nbins, bins_range)
    ghist = np.histogram(image[:,:,1], nbins, bins_range)
    bhist = np.histogram(image[:,:,2], nbins, bins_range)
    #rhist[0] counts in each bin
    #rhist[1] bin edges
    #len(rhist[0])+1 = len(rhist[1])
    
    # Generating bin centers only for plotting...
    #bin_edges = rhist[1]
    #bin_centers = (bin_edges[1:]  + bin_edges[0:len(bin_edges)-1])/2
    
    # Concatenate the histograms into a single feature vector
    # Return the feature vector
    hist_features = np.concatenate((rhist[0], ghist[0], bhist[0]))
    return hist_features
    

# Downsize image and return as feature vector
def bin_spatial(img, color_space, size):
    if color_space != 'RGB':
        string = 'img = cv2.cvtColor(img, cv2.COLOR_RGB2'+color_space+')'
        eval(string)
    img_small = cv2.resize(img, size, interpolation=cv2.INTER_NEAREST)
    
    feature_vec = img_small.ravel()
    return feature_vec



# HOG feature extraction
def hog_features(image, cspace, orient, 
                        pix_per_cell, cell_per_block, hog_channel):
    #Create a list to collect features
    hog_features = []
    # apply color conversion if other than 'RGB'
    if cspace != 'RGB':
        if cspace == 'HSV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        elif cspace == 'LUV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
        elif cspace == 'HLS':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
        elif cspace == 'YUV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
        elif cspace == 'YCrCb':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    else: feature_image = np.copy(image)      

    # Call get_hog_features() with vis=False, feature_vec=True
    if hog_channel == 'ALL':
        
        for channel in range(feature_image.shape[2]):
            hog_features.append((hog(feature_image[:,:,channel], 
                                orient, (pix_per_cell,pix_per_cell), (cell_per_block,cell_per_block), 
                                block_norm= 'L2-Hys', transform_sqrt=True, visualize=False, feature_vector=True)))
        hog_features = np.ravel(hog_features)        
    else:
        hog_features = hog(feature_image[:,:,hog_channel],  orient, (pix_per_cell,pix_per_cell), (cell_per_block,cell_per_block), 
                            block_norm= 'L2-Hys', transform_sqrt=True, visualize=False, feature_vector=True)
    
    # Return list of feature vectors
    return hog_features




# Extract selected features from a list of images
def extract_features(imgs, selectFeatures):
    #PARAMETERS:
    #imgs: List of training image file path
    #selectFeatures: Bool for selection of features for training (Downsized, ColorHist, HOG)
    # OUTPUT: features Vector für alle Bilder in Pfadliste imgs
    
    
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for pimg in imgs:
        # Read in each one by one
        img = cv2.imread(pimg)
        
        
        #Collect selected Features for current training image
        ### SET FEATURE EXTRACTION PARAMETERS HERE
        feat_list = []
        if selectFeatures[0]:
            # 1. Apply bin_spatial() to get spatial downsized image
            #spatial_size: Sets new image dimensions for Downsizing function
            #cspace: Desired image format for feature extraction RGB (Default), HSV,...
            spatial_size = 32
            cspace = 'RGB'
            bin_feat = bin_spatial(img, cspace, (spatial_size,spatial_size))
            feat_list.append(bin_feat)
        if selectFeatures[1]:
            # 2. Apply color_hist() to get color histogram features
            #hist_bins: image sliding window for histogram values
            #hist_range: histogram x axis
            #cspace: desired color space of image
            hist_bins = 32
            hist_range = (0, 255)      
            cspace = 'RGB'
            col_feat = color_hist(img, hist_bins, hist_range, cspace)
            feat_list.append(col_feat)
        if selectFeatures[2]:
            # 3. Apply get_hog_features() to get gradient features
            #colorspace: desired color space of image
            #orient: number of gradient directions extracted per block
            #pix_per_cell: cell size for combined gradient detection (pix,pix)
            #cell_per_block: cells used for block intensity normalization
            colorspace = 'HSV' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
            orient = 9
            pix_per_cell = 8
            cell_per_block = 2
            hog_channel = 2 # Can be 0, 1, 2, or "ALL" (from colorspace)
            hog_feat = hog_features(img, colorspace, orient, pix_per_cell, cell_per_block, hog_channel)
            feat_list.append(hog_feat)
            
        # Append selected features to the features list
        features.append(np.concatenate(feat_list))
    # Return list of feature vectors
    return features




def train_classifier(filename, hog=True, colorhisto=True, downsized=True):
    #PARAMETERS:
    #filename: Location to save trained model
    #hog: Use HOG gradient features?
    #colorhisto: Use color histogram?
    #downsized: Use downsized image as feature?
    #OUTPUT: trained model in 'filename.p'
    
    # Read in car and non-car images
    images_cars = glob.glob('./vehicles/*.png')
    images_notcars = glob.glob('./non-vehicles/*.png')
    cars = []
    notcars = []
    for image in images_cars:
        cars.append(image)
    for image in images_notcars:
        notcars.append(image)
    
    # Extract features for car and noncar images
    selectFeatures = [downsized, colorhisto, hog]
    car_features = extract_features(cars, selectFeatures)
    notcar_features = extract_features(notcars, selectFeatures)
    
    # Create an array stack of feature vectors
    X = np.vstack((car_features, notcar_features)).astype(np.float64)
    
    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))
    
    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=rand_state)
        
    # Fit a per-column scaler only on the training data
    X_scaler = StandardScaler().fit(X_train)
    # Apply the scaler to X_train and X_test
    X_train = X_scaler.transform(X_train)
    X_test = X_scaler.transform(X_test)
    
    
    #CLASSIFIER
    #print('Using spatial binning of:',spatial, 'and', histbin,'histogram bins')
    print('Feature vector length:', len(X_train[0]))
    
    # Use a linear SVC 
    svc = LinearSVC()
    # Check the training time for the SVC
    t=time.time()
    svc.fit(X_train, y_train)
    t2 = time.time()
    print(round(t2-t, 2), 'Seconds to train SVC...')
    
    # Check the score of the SVC
    print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
    # Check the prediction time for a single sample
    t=time.time()
    n_predict = 10
    print('My SVC predicts:     ', svc.predict(X_test[0:n_predict]))
    print('For these',n_predict, 'labels: ', y_test[0:n_predict])
    t2 = time.time()
    print(round(t2-t, 5), 'Seconds to predict', n_predict,'labels with SVC')
    
    pickle.dump(svc, open(filename, 'wb'))
    print('Trained SVC Model was saved successfully in file '+filename)
    return


train_classifier('svcModel_small', hog=True, colorhisto=True ,downsized=False)
