# -*- coding: utf-8 -*-
"""
Created on Sat May 30 12:28:41 2015

@author: andy
"""

import matplotlib.pyplot as plt
import caffe
import scipy as sp
import scipy.ndimage
import seaborn as sns
import skimage.morphology

import caffe_tools as ct

def show_image(image, fig_and_axes=None, **kwargs):
    fig, axes = plt.subplots(1,1) if fig_and_axes is None else fig_and_axes
    axes.imshow(image, aspect='equal', cmap=plt.cm.Blues, interpolation='nearest', **kwargs)  
    axes.set_xticks([])
    axes.set_yticks([])
        
    return fig, axes
    
def visualize_filters(layername, model_file, pretrained, i=0):
    net = caffe.Net(model_file, pretrained, caffe.TEST)
    filters = caffe.io.blobproto_to_array(net.params[layername][i])
    
    fig, axes = plt.subplots(4, 4)
    for f, ax in zip(filters, axes.flat):
        ax.imshow(f[0], interpolation='nearest', cmap=plt.cm.Blues, vmin=filters.min(), vmax=filters.max())

    return filters
    
def show_hough_lines(im, angles, dists):
    fig, ax = show_image(im)
    for angle, dist in zip(angles, dists):
        xs = sp.array([0, im.shape[1]])
        ys = -xs/sp.tan(angle) + dist/sp.sin(angle)
    
        ax.plot(xs, ys, 'r')
    
    ax.set_xlim(0, im.shape[1])
    ax.set_ylim(im.shape[0], 0)
    
    return fig, ax    

def find_borders(array, threshold=0.65):
    boolean_array = (array > threshold)
    structure = skimage.morphology.disk(1)
    boolean_borders = ~sp.ndimage.binary_erosion(boolean_array, structure) & boolean_array
    
    return boolean_borders

def one_channel_to_color(im):
    pixels = plt.cm.Blues(im.ravel())
    color_im = pixels.reshape((im.shape[0], im.shape[1], 4))[:, :, :3]
    color_im = (255*color_im).astype(sp.uint8)
    
    return color_im
    
def two_channel_to_color(im):
    lower = sp.percentile(im, 5)
    upper = sp.percentile(im, 98)   
    
    channel_0 = sp.clip((im[:, :, 0] - lower)/(upper - lower), 0, 1)
    channel_2 = sp.clip((im[:, :, 1] - lower)/(upper - lower), 0, 1)
    channel_1 = ((channel_0 + channel_2)/2.)
    
    im = sp.array((channel_0, channel_1, channel_2))
    im = sp.rollaxis(im, 0, 3)
    
    im = (255*im).astype(sp.uint8)    
    
    return im

def plot_borders(im, array, threshold=0.65, alpha=0.7, **kwargs):
    borders = find_borders(array)
    im_with_borders = two_channel_to_color(im)
    im_with_borders[borders, :] = (1-alpha)*im_with_borders[borders, :] + alpha*sp.array([255, 0, 0])
    
    return ct.show_image(im_with_borders, **kwargs) 