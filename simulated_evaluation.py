# -*- coding: utf-8 -*-
"""
Created on Sat May 30 14:45:03 2015

@author: andy
"""
import scipy as sp

from simulated_tools import get_simulated_im, get_simulated_ims
from caffe_tools import create_classifier, score_image, make_score_array

TEST_LOW_IDS = ['exp_low ({0})'.format(i) for i in range(25, 50)] 
TEST_GOOD_IDS = ['exp_good ({0})'.format(i) for i in range(1, 50)]

MODEL_FILE = r'caffe_definitions/simulated_deploy.prototxt'
PRETRAINED = 'caffe_definitions/simulated_7_160k.caffemodel'

def get_data(file_id, channel=0):
    im, truth = get_simulated_im(file_id, channel=channel)
    window_centers = sp.load('temporary/scores_{0}/{1}_window_centers.npy'.format(channel, file_id))
    score_list = sp.load('temporary/scores_{0}/{1}_score_list.npy'.format(channel, file_id))

    return im, truth, window_centers - 20, score_list

def calculate_error_rate(file_ids, threshold=0.65):
    error_rates = []
    
    for file_id in file_ids:
        im, truth, centers, scores = get_data(file_id, channel=-1)
        predictions = make_score_array(scores[:, 1], centers, truth.shape)
        incorrect = (predictions > threshold) != (truth > 0.5)
        
        error_rates.append(incorrect.sum()/float(truth.size))
        
    return sp.median(error_rates)

def calculate_discrepency_distance(file_ids, threshold=0.65):
    discrepency_distances = []        
    
    for file_id in file_ids:
        im, truth, centers, scores = get_data(file_id, channel=-1)
        predictions = make_score_array(scores[:, 1], centers, truth.shape)

        false_positives = (predictions > threshold) & (truth < 0.5)
        false_dt = sp.ndimage.distance_transform_edt(truth < 0.5)        
        false_discrepency = ((false_dt*false_positives)**2).sum()
        
        false_negatives = (predictions <= threshold) & (truth >= 0.5)
        true_dt = sp.ndimage.distance_transform_edt(truth >= 0.5)
        true_discrepency = ((true_dt*false_negatives)**2).sum()
        
        discrepency_distance = sp.sqrt(false_discrepency + true_discrepency)/float(truth.size)
        discrepency_distances.append(discrepency_distance)
        
    return sp.median(discrepency_distances)

def score_images(channel=-1):
    classifier = create_classifier(MODEL_FILE, PRETRAINED)
    ims, _ = get_simulated_ims(channel=channel)
    for i, (file_id, im) in enumerate(ims.items()): 
        print('Processing file {0}, {1} of {2}'.format(file_id, i+1, len(ims)))
        window_centers, score_list = score_image(im, classifier, width=41)
        sp.save('temporary/scores_{0}/{1}_score_list'.format(channel, file_id), score_list)
        sp.save('temporary/scores_{0}/{1}_window_centers'.format(channel, file_id), window_centers)    
