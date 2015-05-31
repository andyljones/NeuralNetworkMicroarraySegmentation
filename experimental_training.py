# -*- coding: utf-8 -*-
"""
This module uses a set of hand-label experimental microarray images to produce a LMDB file that can be used to train a
Caffe neural network. The main functions of interest are ``visualize_image_for_hand_labelling`` and 
``make_training_files``.

To build a training set, first use ``visualize_image_for_hand_labelling`` to save out a human-interpretable image of a 
microarray file. Then use image editing software to color the image according to the ``GOOD_COLOR``, ``DAMAGED_COLOR``,
etc attributes. If you put the labelled images into the ``LABELS_FOLDER`` and add the corresponding file IDs 
``LABELLED_FILE_IDS``, then calling ``make_training_files`` will construct a pair of LMDB databases that can be used to 
train and test a Caffe classifier.
"""

import scipy as sp
import tifffile
import os

from experimental_tools import get_bounded_im, get_benchmark_im, WINDOW_WIDTH
from visualization_tools import two_channel_to_color
from caffe_tools import fill_database

"""The colours used to hand-label different kinds of spots"""
GOOD_COLOR = [255, 0, 0] # red, meant for pixels inside whole spots.
DAMAGED_COLOR = [255, 255, 0] # yellow, meant for pixels inside damaged spots.
MISSING_COLOR = [255, 0, 255] # magenta, meant for pixels at the approximate location of a spot that's extremely faint
LABELLED_AREA_COLOR = [0, 255, 0] # green, meant for marking the border of a block of labelled spots
UNLABELLED_SPOTTED_AREA_COLOR = [0, 0, 255] # blue, meant for marking the border of all blocks of spots that haven't been labelled.

"""The folder hand-labelled images are expected to be in"""
LABELS_FOLDER = 'sources/labels'

"""The mapping from types of pixels to classifier labels"""
LABEL_ENUM = {'inside': 1, 
              'outside': 0, 
              'inside_damaged': 1,
              'outside_damaged': 0, 
              'block_border': 0,
              'between': 0}

"""IDs of the hand-labelled images"""
LABELLED_FILE_IDS = ['3-12_pmt100', '3-16_pmt100']
    
    
def visualize_image_for_hand_labelling(file_id):
    """Saves out a visualization of the image ``file_id`` that's appropriate for hand-labelling"""
    im = get_bounded_im(file_id, channel=-1)
    im = two_channel_to_color(im)
    
    target_path = os.path.join(LABELS_FOLDER, '{}_corrected.tif'.format(file_id))
    tifffile.imsave(target_path, im)

def equal_color_mask(im, color):
    """Returns a mask indicating which pixels in ``im`` are ``color``"""
    return reduce(sp.logical_and, [im[:, :, i] == color[i] for i in range(3)])

def get_hand_labels(file_id):
    """Returns the spots and areas that have been hand labelled for ``file_id``"""
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
    """Returns a mask indicating which pixels lie inside a good or damaged spot"""
    return spots['good'] | spots['damaged']
    
def make_outside_mask(spots, areas):
    """Returns a mask indicating which pixels lie outside good spots, damaged spots, or an area marked as containing 
    spots"""
    inside = make_inside_mask(spots)
    near_inside = sp.ndimage.binary_dilation(inside, structure=sp.ones((3,3))) 
    
    return ~(near_inside | areas['unlabelled'])
    
def make_inside_damaged_mask(spots):
    """Returns a mask indicating which pixels lie inside damaged spots"""
    return spots['damaged']
    
def make_outside_damaged_mask(spots, areas):
    """Returns a mask indicating which pixels lie just outside damaged spots"""
    outside = make_outside_mask(spots, areas)
    inside_damaged = make_inside_damaged_mask(spots)
    near_damaged = sp.ndimage.binary_dilation(inside_damaged, structure=sp.ones((3,3)), iterations=8)
    outside_near_damaged = near_damaged & outside
    
    return outside_near_damaged
    
def make_block_border_mask(spots, areas):
    """Returns a mask indicating which pixels lie just outside a block of spots"""
    inside = make_inside_mask(spots)
    outside = make_outside_mask(spots, areas)
    very_near_inside = sp.ndimage.binary_dilation(inside, structure=sp.ones((3,3)), iterations=8)
    near_inside = sp.ndimage.binary_dilation(inside, structure=sp.ones((3,3)), iterations=32)
    return near_inside & ~very_near_inside & outside

def make_between_mask(spots, areas):
    """Returns a mask indicating which pixels lie between two spots"""
    inside = make_inside_mask(spots)
    outside = make_outside_mask(spots, areas)   
    
    near_inside = sp.ndimage.binary_dilation(inside, structure=sp.ones((3,3)), iterations=8)
    
    return near_inside & outside

def find_centers(spots, areas, border_width, im_num=0):
    """Returns a dict of arrays, one for each pixel type. The arrays are compatible with caffe_tools.fill_database.

    The last row of each array is equal to ``im_num``, indicating which image those centers were created from.
    """
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

def find_centers_from_ims(file_ids, width):
    """Uses the images at ``file_ids`` to create a dict of arrays indexed by pixel type. The arrays are compatible with 
    caffe_tools.fill_database."""
    centers = []
    for i, file_id in enumerate(file_ids):
        spots, areas = get_hand_labels(file_id)
        centers.append(find_centers(spots, areas, width/2, im_num=i))
    
    result = {}
    for name in centers[0]:
        result[name] = sp.concatenate([cs[name] for cs in centers], 1)
        
    return result    
    
def make_labelled_sets(centers, test_split=0.1):
    """Uses a dict of arrays like those created by ``find_centers_from_ims`` to build test and training sets for training 
    a Caffe model to distinguish different types of pixel. The arrays returned are centers and labels compatible with 
    caffe_tools.fill_database"""
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
            
def create_caffe_input_files(file_ids, width):    
    """Creates LMDB databases containing training and test sets derived from the hand-labelled ``file_ids``. ``width``
    is the size of the windows to use.
    
    The databases can be found in the ``temporary`` directory."""    
    ims = [get_benchmark_im(file_id) for file_id in file_ids]
    ims = [(im - im.mean())/im.std() for im in ims]
    
    centers = find_centers_from_ims(file_ids, width)
    training_centers, training_labels, test_centers, test_labels = make_labelled_sets(centers)

    fill_database('temporary/train_experimental.db', ims, training_centers, training_labels, width)
    fill_database('temporary/test_experimental.db', ims, test_centers, test_labels, width)

def make_training_files():
    """Use the hand-labels corresponding to ``LABELLED_FILE_IDS`` to create LMDB databases containing training and test
    sets for a Caffe neural network."""
    create_caffe_input_files(LABELLED_FILE_IDS, WINDOW_WIDTH)