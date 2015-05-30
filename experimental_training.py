# -*- coding: utf-8 -*-
"""
Created on Sat May 30 12:15:32 2015

@author: andy
"""

import scipy as sp
import tifffile
import os

from experimental_tools import get_bounded_im, get_benchmark_im
from visualization_tools import two_channel_to_color
from caffe_tools import fill_database

GOOD_COLOR = [255, 0, 0]
DAMAGED_COLOR = [255, 255, 0]
MISSING_COLOR = [255, 0, 255]
LABELLED_AREA_COLOR = [0, 255, 0]
UNLABELLED_SPOTTED_AREA_COLOR = [0, 0, 255]

LABELS_FOLDER = 'sources/labels'

LABEL_ENUM = {'inside': 1, 
              'outside': 0, 
              'inside_damaged': 2,
              'outside_damaged': 0, 
              'block_border': 0,
              'between': 0}


labelled_ids = ['3-12_pmt100', '3-16_pmt100']

def normalize_channel(im):
    lower = sp.percentile(im, 5)
    upper = sp.percentile(im, 95)
    
    normalized = sp.clip((im - lower)/(upper - lower), 0, 1)
    
    return normalized 
    
def correct_image_for_gimp(file_id):
    im = get_bounded_im(file_id, channel=-1)
    im = two_channel_to_color(im)
    
    target_path = 'sources/labels/{0}_corrected.tif'.format(file_id)
    tifffile.imsave(target_path, im)
    
    return im

def equal_color_mask(im, color):
    return reduce(sp.logical_and, [im[:, :, i] == color[i] for i in range(3)])

def get_hand_labels(file_id):
    labels_filename = file_id + '_corrected_labelled.png' 
    labels_path = os.path.join(LABELS_FOLDER, labels_filename)
    labels = sp.ndimage.imread(labels_path)
    
    spots = {}    
    spots['good'] = equal_color_mask(labels, GOOD_COLOR)
    spots['damaged'] = equal_color_mask(labels, DAMAGED_COLOR)
    spots['missing'] = equal_color_mask(labels, MISSING_COLOR)    
    
    labelled_area_outlines = equal_color_mask(labels, LABELLED_AREA_COLOR)
    unlabelled_spotted_area_outlines = equal_color_mask(labels, UNLABELLED_SPOTTED_AREA_COLOR)

    areas = {}
    areas['labelled'] = sp.ndimage.binary_fill_holes(labelled_area_outlines)
    areas['unlabelled'] = sp.ndimage.binary_fill_holes(unlabelled_spotted_area_outlines)

    return spots, areas

def make_inside_mask(spots):
    return spots['good'] | spots['damaged']
    
def make_outside_mask(spots, areas):
    inside = make_inside_mask(spots)
    near_inside = sp.ndimage.binary_dilation(inside, structure=sp.ones((3,3))) 
    
    return ~(near_inside | areas['unlabelled'])
    
def make_inside_damaged_mask(spots):
    return spots['damaged']
    
def make_outside_damaged_mask(spots, areas):
    outside = make_outside_mask(spots, areas)
    inside_damaged = make_inside_damaged_mask(spots)
    near_damaged = sp.ndimage.binary_dilation(inside_damaged, structure=sp.ones((3,3)), iterations=8)
    outside_near_damaged = near_damaged & outside
    
    return outside_near_damaged
    
def make_block_border_mask(spots, areas):
    inside = make_inside_mask(spots)
    outside = make_outside_mask(spots, areas)
    very_near_inside = sp.ndimage.binary_dilation(inside, structure=sp.ones((3,3)), iterations=8)
    near_inside = sp.ndimage.binary_dilation(inside, structure=sp.ones((3,3)), iterations=32)
    return near_inside & ~very_near_inside & outside

def make_between_mask(spots, areas):
    inside = make_inside_mask(spots)
    outside = make_outside_mask(spots, areas)   
    
    near_inside = sp.ndimage.binary_dilation(inside, structure=sp.ones((3,3)), iterations=8)
    
    return near_inside & outside

def find_centers(spots, areas, border_width=32, im_num=0):
    indices = sp.indices(spots['good'].shape)
    indices = sp.concatenate([indices, im_num*sp.ones((1, indices.shape[1], indices.shape[2]), dtype=int)], 0)

    inside_border = sp.zeros(spots['good'].shape, dtype=bool)
    inside_border[border_width:-border_width, border_width:-border_width] = True
    
    centers = {}
    centers['inside'] = indices[:, make_inside_mask(spots) & inside_border]
    centers['outside'] = indices[:, make_outside_mask(spots, areas) & inside_border]
    centers['inside_damaged'] = indices[:, make_inside_damaged_mask(spots) & inside_border]
    centers['outside_damaged'] = indices[:, make_outside_damaged_mask(spots, areas) & inside_border]
    centers['block_border'] = indices[:, make_block_border_mask(spots, areas) & inside_border]
    centers['between'] = indices[:, make_between_mask(spots, areas) & inside_border]
    
    return centers

def find_centers_from_ims(file_ids):
    
    centers = []
    for i, file_id in enumerate(file_ids):
        spots, areas = get_hand_labels(file_id)
        centers.append(find_centers(spots, areas, im_num=i))
    
    result = {}
    for name in centers[0]:
        result[name] = sp.concatenate([cs[name] for cs in centers], 1)
        
    return result    
    
def make_labelled_sets(centers, test_split=0.1):
    counts = {'inside': 100e3, 'outside': 50e3, 'inside_damaged': 100e3, 'outside_damaged': 50e3, 'block_border': 50e3, 'between': 50e3}
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
            
def create_caffe_input_file(file_ids, channel=0, width=61):    
    ims = [get_benchmark_im(file_id)[channel] for file_id in file_ids]
    ims = [(im - im.mean())/im.std() for im in ims]
    
    centers = find_centers_from_ims(file_ids)
    training_centers, training_labels, test_centers, test_labels = make_labelled_sets(centers)

    fill_database('temporary/train_experimental.db', ims, training_centers, training_labels, width)
    fill_database('temporary/test_experimental.db', ims, test_centers, test_labels, width)
