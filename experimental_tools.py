# -*- coding: utf-8 -*-
"""
Created on Sat May 30 12:15:52 2015

@author: andy
"""

import tifffile
import scipy as sp
import os

IMAGE_FOLDER = 'sources/experimental'
WINDOW_WIDTH = 61
boundaries = [slice(650,-150), slice(1900,-500)]

def get_benchmark_im(file_id):
    filepath = os.path.join(IMAGE_FOLDER, file_id + '.tif')
    return sp.rollaxis(sp.log(tifffile.imread(filepath)), 0, 3)

def get_bounded_im(file_id):
    im = get_benchmark_im(file_id)
    
    return im[:, boundaries[0], boundaries[1]]

def get_bounded_ims():
    filenames = [f for f in os.listdir(IMAGE_FOLDER) if os.path.splitext(f)[1] == '.tif']
    file_ids = [filename.split('.tif')[0] for filename in filenames]
    return {file_id: get_bounded_im(file_id) for file_id in file_ids}