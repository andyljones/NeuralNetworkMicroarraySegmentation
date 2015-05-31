# -*- coding: utf-8 -*-
"""
Created on Sat May 30 14:44:52 2015

@author: andy
"""

import os
import scipy as sp

"""The folder containing the simulated data files"""
SIMULATED_BENCHMARK_FOLDER = 'sources/simulated'

"""The width of the window to use when classifying a pixel as foreground or background"""
WINDOW_WIDTH = 41

def get_simulated_im(file_id):
    """Gets the image and ground truth corresponding to the simulated data file ``file_id``"""
    filepath = os.path.join(SIMULATED_BENCHMARK_FOLDER, file_id + '.mat')
    mat = sp.io.loadmat(filepath)
    # The mat files consist of an image array, a ground truth array and then a bunch of arrays giving the centers and 
    # sizes of each spot pre-corruption.    
    
    im = mat['images'][0, 0][:, :, :2]
    
    truth = mat['meta'][0, 0][0][0, 0][:, :, 0]
    return im, truth

def get_simulated_ims():
    """Gets the images and ground truths for all the simulated data files."""
    filenames = [f for f in os.listdir(SIMULATED_BENCHMARK_FOLDER) if os.path.splitext(f)[1] == '.mat']
    file_ids = [filename.split('.mat')[0] for filename in filenames]
    ims = {file_id: get_simulated_im(file_id)[0] for file_id in file_ids}    
    truths = {file_id: get_simulated_im(file_id)[1] for file_id in file_ids}    
    
    return ims, truths
    