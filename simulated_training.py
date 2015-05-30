# -*- coding: utf-8 -*-
"""
Created on Sat May 30 14:45:10 2015

@author: andy
"""
import scipy as sp

from simulated_tools import get_simulated_im
from caffe_tools import fill_database

TRAINING_IDS = ['exp_low ({0})'.format(i) for i in range(1, 25)]

LABEL_ENUM = {'inside': 1, 
              'outside': 0, 
              'inside_damaged': 1, 
              'outside_damaged': 0, 
              'block_border': 0, 
              'between': 0}

def make_damaged_spot_mask(truth):
    damaged_pixel = (0.75 < truth) & (truth < 1)
    damaged_area = sp.ndimage.binary_closing(damaged_pixel, structure=sp.ones((3, 3)))
    damaged_spot = damaged_area & (0.75 < truth)
    
    return damaged_spot
    
def make_outside_near_damaged_spot_mask(truth):
    damaged_spot = make_damaged_spot_mask(truth)
    near_damaged_spot = sp.ndimage.binary_dilation(damaged_spot, structure=sp.ones((3,3)), iterations=5)
    outside_near_damaged_spot = near_damaged_spot & (truth < 0.25)
    
    return outside_near_damaged_spot

def make_block_border_mask(truth):
    very_near_block = sp.ndimage.binary_dilation(0.75 < truth , structure=sp.ones((3,3)), iterations=3)
    near_block = sp.ndimage.binary_dilation(0.75 < truth , structure=sp.ones((3,3)), iterations=15)
    block_border = near_block & ~very_near_block
    
    return block_border
    
def make_between_spot_mask(truth):
    near_spot = sp.ndimage.binary_dilation(0.75 < truth, structure=sp.ones((3, 3)), iterations=4)
    outside_near_spot = near_spot & (truth < 0.25)
    
    return outside_near_spot

def get_centers_single_image(truth, im_no, border=20):
    indices = sp.indices(truth.shape)
    im_nos = im_no*sp.ones((1, truth.shape[0], truth.shape[1]), dtype=int)
    indices = sp.concatenate((indices, im_nos))
    
    away_from_border = sp.zeros(truth.shape, dtype=bool)
    away_from_border[border:-border, border:-border] = True
    
    results = {
    'inside': indices[:, (0.75 < truth) & away_from_border]
    ,'outside': indices[:, (truth < 0.25) & away_from_border]
    ,'inside_damaged' : indices[:, make_damaged_spot_mask(truth) & away_from_border]
    ,'outside_damaged': indices[:, make_outside_near_damaged_spot_mask(truth) & away_from_border]
    ,'block_border': indices[:, make_block_border_mask(truth) & away_from_border]
    ,'between': indices[:, make_between_spot_mask(truth) & away_from_border]
    }
    
    return results
    
def get_centers(truths, channel=0, border=20):
    centers = []
    for i, truth in enumerate(truths):
        centers.append(get_centers_single_image(truth, i, border=border))
    
    result = {}
    for name in centers[0]:
        result[name] = sp.concatenate([cs[name] for cs in centers], 1)
        
    return result
    
def make_labelled_sets(centers, test_split=0.1):
    counts = {'inside': 2e5, 'outside': 1e5, 'inside_damaged': 2e5, 'outside_damaged': 1e5, 'block_border': 1e5, 'between': 1e5}
    choices = {name: sp.random.choice(sp.arange(centers[name].shape[1]), counts[name]) for name in centers}
    center_sets = {name: centers[name][:, choices[name]] for name in centers}
    label_sets = {name: sp.repeat(LABEL_ENUM[name], counts[name]) for name in centers}
    
    center_set = sp.concatenate([center_sets[name] for name in centers], 1)
    label_set = sp.concatenate([label_sets[name] for name in centers])

    order = sp.random.permutation(sp.arange(center_set.shape[1]))
    ordered_centers = center_set[:, order]
    ordered_labels = label_set[order]
    
    n_training = int((1-test_split)*center_set.shape[1])
    training_centers = ordered_centers[:, :n_training]
    training_labels = ordered_labels[:n_training]
    test_centers = ordered_centers[:, n_training:]
    test_labels = ordered_labels[n_training:]
    
    return training_centers, training_labels, test_centers, test_labels
    
def create_caffe_input_file(file_ids, width=41):    
    im_padding = ((width/2, width/2), (width/2, width/2), (0, 0))
    ims = [get_simulated_im(file_id, channel=-1)[0] for file_id in file_ids]
    ims = [(im - im.mean())/im.std() for im in ims]
    ims = [sp.pad(im, im_padding, mode='reflect') for im in ims]
    
    truth_padding =  ((width/2, width/2), (width/2, width/2))
    truths = [get_simulated_im(file_id)[1] for file_id in file_ids]
    truths = [sp.pad(truth, truth_padding, mode='reflect') for truth in truths]
    
    centers = get_centers(truths, width/2)
    training_centers, training_labels, test_centers, test_labels = make_labelled_sets(centers)

    fill_database('temporary/train_simulated.db', ims, training_centers, training_labels, width)
    fill_database('temporary/test_simulated.db', ims, test_centers, test_labels, width)