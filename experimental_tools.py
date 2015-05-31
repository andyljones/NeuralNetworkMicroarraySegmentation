# -*- coding: utf-8 -*-
"""
These are tools used by both the training and evaluation components of the experimental pipeline.
"""

import tifffile
import scipy as sp
import os

"""The folder containing the experimental images"""
IMAGE_FOLDER = 'sources/experimental'

"""The width of the window to use when classifying a pixel as foreground or background"""
WINDOW_WIDTH = 61

"""The slice of the experimental images that actually contains spots"""
boundaries = [slice(650,-150), slice(1900,-500)]

def get_benchmark_im(file_id):
    """Gets the experimental image associated with ``file_id``"""
    filepath = os.path.join(IMAGE_FOLDER, file_id + '.tif')
    return sp.rollaxis(sp.log(tifffile.imread(filepath)), 0, 3)

def get_bounded_im(file_id):
    """Gets the experimental image associated with ``file_id``, then restricts it to the area with spots in"""
    im = get_benchmark_im(file_id)
    
    return im[boundaries[0], boundaries[1]]

def get_bounded_ims():
    """Gets all the experimental images, and restricts each to the area with spots in"""
    filenames = [f for f in os.listdir(IMAGE_FOLDER) if os.path.splitext(f)[1] == '.tif']
    file_ids = [filename.split('.tif')[0] for filename in filenames]
    return {file_id: get_bounded_im(file_id) for file_id in file_ids}