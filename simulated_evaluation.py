# -*- coding: utf-8 -*-
"""
Created on Sat May 30 14:45:03 2015

@author: andy
"""
import scipy as sp
import h5py

from simulated_tools import get_simulated_im, get_simulated_ims
from caffe_tools import create_classifier, score_images

TEST_LOW_IDS = ['exp_low ({0})'.format(i) for i in range(25, 50)] 
TEST_GOOD_IDS = ['exp_good ({0})'.format(i) for i in range(1, 50)]

DEFINITION_PATH = 'sources/definitions/simulated_deploy.prototxt'
MODEL_PATH = 'temporary/models/simulated_iter_20000.caffemodel'

SCORES_PATH = 'temporary/scores/simulated_scores.hdf5' 
WINDOW_WIDTH = 41

def score_simulated_images():
    classifier = create_classifier(DEFINITION_PATH, MODEL_PATH)
    ims, _ = get_simulated_ims()
    with h5py.File(SCORES_PATH, 'w-') as h5file:
        score_images(h5file, ims, classifier, WINDOW_WIDTH)          

def get_data(file_id):
    im, truth = get_simulated_im(file_id)
    with h5py.File(SCORES_PATH, 'r') as h5file:
        score_array = h5file[file_id]        

    return im, truth, score_array

def calculate_error_rate(file_ids, threshold=0.65):
    error_rates = []
    
    for file_id in file_ids:
        im, truth, predictions = get_data(file_id)
        incorrect = (predictions > threshold) != (truth > 0.5)
        
        error_rates.append(incorrect.sum()/float(truth.size))
        
    return sp.median(error_rates)

def calculate_discrepency_distance(file_ids, threshold=0.65):
    discrepency_distances = []        
    
    for file_id in file_ids:
        im, truth, predictions = get_data(file_id)

        false_positives = (predictions > threshold) & (truth < 0.5)
        false_dt = sp.ndimage.distance_transform_edt(truth < 0.5)        
        false_discrepency = ((false_dt*false_positives)**2).sum()
        
        false_negatives = (predictions <= threshold) & (truth >= 0.5)
        true_dt = sp.ndimage.distance_transform_edt(truth >= 0.5)
        true_discrepency = ((true_dt*false_negatives)**2).sum()
        
        discrepency_distance = sp.sqrt(false_discrepency + true_discrepency)/float(truth.size)
        discrepency_distances.append(discrepency_distance)
        
    return sp.median(discrepency_distances) 
