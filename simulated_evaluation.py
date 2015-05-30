# -*- coding: utf-8 -*-
"""
Created on Sat May 30 14:45:03 2015

@author: andy
"""
import scipy as sp
import h5py
import logging

from simulated_tools import get_simulated_im, get_simulated_ims
from caffe_tools import create_classifier, score_image, make_score_array

TEST_LOW_IDS = ['exp_low ({0})'.format(i) for i in range(25, 50)] 
TEST_GOOD_IDS = ['exp_good ({0})'.format(i) for i in range(1, 50)]

MODEL_FILE = 'sources/definitions/simulated_deploy.prototxt'
PRETRAINED = 'temporary/models/simulated_iter_20000.caffemodel'

SIMULATED_SCORES_PATH = 'temporary/scores/simulated_scores.hdf5' 

def score_images():
    classifier = create_classifier(MODEL_FILE, PRETRAINED)
    ims, _ = get_simulated_ims()
    with h5py.File(SIMULATED_SCORES_PATH, 'w-') as h5file:
        for i, (file_id, im) in enumerate(ims.items()): 
            logging.info('Processing file {0}, {1} of {2}'.format(file_id, i+1, len(ims)))
            window_centers, score_list = score_image(im, classifier, width=41)
            score_array = make_score_array(window_centers, score_list[:, 1], im.shape[:2])
            h5file[file_id] = score_array            

def get_data(file_id):
    im, truth = get_simulated_im(file_id)
    with h5py.File(SIMULATED_SCORES_PATH, 'r') as h5file:
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
