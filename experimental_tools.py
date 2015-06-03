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