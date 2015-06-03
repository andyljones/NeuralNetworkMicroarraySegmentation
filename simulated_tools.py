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
These are tools used by both the training and evaluation components of the simulated pipeline.
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
    