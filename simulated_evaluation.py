# -*- coding: utf-8 -*-
"""
Copyright (c) 2015, Andrew Jones (andyjones dot ed at gmail dot com)
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of the <organization> nor the
      names of its contributors may be used to endorse or promote products
      derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

"""
This module uses a trained neural network to segment simulated microarray images into foreground and background 
pixels. It can then calculate the error rate and discrepency distances from the results.
"""
import scipy as sp
import h5py

from simulated_tools import get_simulated_im, get_simulated_ims, WINDOW_WIDTH
from caffe_tools import create_classifier, score_images

"""File IDs to be used for testing"""
TEST_LOW_IDS = ['exp_low ({0})'.format(i) for i in range(25, 50)] 
TEST_GOOD_IDS = ['exp_good ({0})'.format(i) for i in range(1, 50)]

"""The path to the file describing the Caffe classifier."""
DEFINITION_PATH = 'sources/definitions/simulated_deploy.prototxt'

"""The path to the file containing the trained Caffe model"""
MODEL_PATH = 'temporary/models/simulated_iter_48000.caffemodel'

"""The path to the file containing the score arrays for each image"""
SCORES_PATH = 'temporary/scores/simulated_scores.hdf5' 

def score_simulated_images():
    """Scores each pixel in each simulated image using a Caffe classifier, and stores the results in a HDF5 file."""
    classifier = create_classifier(DEFINITION_PATH, MODEL_PATH)
    ims, _ = get_simulated_ims()
    score_images(SCORES_PATH, ims, classifier, WINDOW_WIDTH)          

def get_data(file_id):
    """Gets the image, ground truth and the score array associated with ``file_id``"""
    im, truth = get_simulated_im(file_id)
    with h5py.File(SCORES_PATH, 'r') as h5file:
        score_array = h5file[file_id]        

    return im, truth, score_array

def calculate_error_rate(file_ids, threshold=0.65):
    """Calcuates the segmentation error rate when the threshold for predicting a pixel as foreground is set at 
    ``threshold``."""
    error_rates = []
    
    for file_id in file_ids:
        im, truth, predictions = get_data(file_id)
        incorrect = (predictions > threshold) != (truth > 0.5)
        
        error_rates.append(incorrect.sum()/float(truth.size))
        
    return sp.median(error_rates)

def calculate_discrepency_distance(file_ids, threshold=0.65):
    """Calcuates the segmentation discrepency distance when the threshold for predicting a pixel as foreground is set 
    at ``threshold``."""
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
