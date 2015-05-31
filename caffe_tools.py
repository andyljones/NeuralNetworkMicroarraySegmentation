# -*- coding: utf-8 -*-
"""
This is a collection of tools for interacting with Caffe. It's used by both the experimental and simulated pipelines 
"""

import scipy as sp
import caffe
import lmdb
import os
import datetime
import itertools as it
import logging


def get_window(im, center, width):
    """Returns a window on ``im`` centered on ``center`` and with width ``width``"""
    if width % 2 == 0:
        i_slice = slice(center[0]-width/2, center[0]+width/2)
        j_slice = slice(center[1]-width/2, center[1]+width/2)
    else:
        i_slice = slice(center[0]-width/2, center[0]+width/2+1)
        j_slice = slice(center[1]-width/2, center[1]+width/2+1)
        
    return im[i_slice, j_slice]

def randomly_flip(im):
    """Returns a copy of ``im`` randomly flipped horizontally, vertically, or both"""
    if sp.rand() > 0.5:
        im = im[::-1, :]
    if sp.rand() > 0.5:
        im = im[:, ::-1]
        
    return im    
    
def randomly_rotate(im):    
    """Returns a copy of ``im`` randomly rotated through 0, 90, 180 or 270 degrees"""
    k = sp.random.randint(0, 4)
    return sp.ndimage.interpolation.rotate(im, 90*k)
    
def make_datum(im, center, label, width):
    """Creates a Caffe datum object from a window on ``im`` with the given label"""
    window = get_window(im, center, width)
    window = randomly_flip(window)
    window = randomly_rotate(window)
    window = sp.rollaxis(window, 2, 0)
    datum = caffe.io.array_to_datum(window.astype(float), label)
    return datum

def progress_report(count_so_far, total_count, start_time):
    """Returns a string reporting the progress towards ``total_count`` since ``start_time``"""
    current_time = datetime.datetime.now()
    time_passed = current_time - start_time
    time_per = time_passed/count_so_far
    time_to_go = (total_count - count_so_far)*time_per
    
    return 'Processed {0} of {1} window centers so far in {2}. Still to go: {3}'.format(
                count_so_far, 
                total_count, 
                str(time_passed), 
                str(time_to_go))

def fill_database(name, ims, centers, labels, width):
    """
    Creates and fills a LMDB database of labelled windows that can be used to train a Caffe model.
    
    Parameters
    ----------
    name : string
        Name (path) of the database to be created.
    ims : sequence
        A sequence of (two-channel microarray) images.
    centers : array_like
        An int array of shape ``(3, n)``. If ``i, j, k = centers[:, m]``, then the ``m``th window is centered at `(i, j)` 
        in ``ims[k]``.
    labels : sequence
        The labels for each window; ``labels[m]`` is the label for the window described by ``centers[:, m]``.
    width : int
        The width of the windows.
        
    Raises
    ------
    ValueError
        If the database ``name`` already exists.
    """
    start_time = datetime.datetime.now()
    
    if os.path.exists(name): raise ValueError('Database {0} already exists'.format(name))     
    
    env = lmdb.open(name, map_size=1e12)
    with env.begin(write=True) as txn:
        for i, (center, label) in enumerate(zip(centers.T, labels)):
            im_no = center[2]
            datum = make_datum(ims[im_no], center[:2], label, width)
            key = '{0}_{1}-{2}-{3}'.format(i, im_no, center[0], center[1])
            txn.put(key, datum.SerializeToString())
            
            if i % 500 == 0: logging.info(progress_report(i+1, centers.shape[1], start_time))
                
    env.close()
            
def load_from_db(name, key):
    """Gets the array and label corresponding to the specified ``key`` from database ``name``"""
    env = lmdb.open(name)
    with env.begin(write=False) as txn:
        s = txn.get(key)
        datum = caffe.proto.caffe_pb2.Datum()
        datum.ParseFromString(s)
        
        arr = caffe.io.datum_to_array(datum)
        
        return arr, datum.label

def make_score_array(score_list, window_centers, shape):
    """
    Converts a list of scores and their locations into an array.

    Parameters
    ----------
    score_list : sequence
        A n-length sequence of floats, typically between 0 and 1.
    window_centers : array_like
        A (n, 2)-shaped int array giving the indices of the scores in ``score_list``.
    shape : 
        The shape of the array to be returned.
    """
    score_array = sp.zeros(shape)
    score_array[window_centers[:, 0], window_centers[:, 1]] = score_list

    return score_array
    
def get_window_centers(im, width):
    """Returns a list of all indices more than ``width`` distant from the boundaries of ``im``"""
    boundary = int(sp.ceil(width/2))
    yslice = slice(boundary, im.shape[0]-boundary)
    xslice = slice(boundary, im.shape[1]-boundary)
    indices = sp.mgrid[yslice, xslice]
    indices = sp.rollaxis(indices, 0, 3).reshape((-1, 2))
    
    return indices

def window_generator(im, window_centers, width):
    """Iterates over ``window_centers`` and yields a window of ``width`` onto ``im`` centered on each element."""
    for window_center in window_centers:
        window = get_window(im, window_center, width)
        yield window
        
    return 

def score_window_list(window_list, model):
    """Scores each window in ``window_list`` using the Caffe classifier ``model``. Each window has all its rotations and
    reflections scored, and the median score is used."""
    unflipped = sp.array(window_list)
    flipped_ud = unflipped[:, ::-1]

    flipped = [unflipped, flipped_ud]
    rotated = [sp.ndimage.interpolation.rotate(f, 90*k, axes=(1, 2), order=0) for f in flipped for k in range(3)]

    predictions = [model.predict(ws, oversample=False) for ws in rotated] 
    return sp.median(sp.array(predictions), 0)

def score_windows(window_generator, model, total_count):
    """Scores each window yielded by ``window_generator`` using the Caffe classifier ``model``"""
    start_time = datetime.datetime.now()
    count_so_far = 0    
    
    results = []    
    while True:
        windows = list(it.islice(window_generator, 4096))
        if not windows:
            break
        
        result = score_window_list(windows, model)
        results.append(result)
        
        count_so_far = count_so_far + len(result)
        
        if total_count:
            logging.info(progress_report(count_so_far, total_count, start_time))
        else:
            logging.info('Processed {0} window_centers'.format(count_so_far))

        
    results = sp.concatenate(results)
    
    return results
    
def score_image(im, model, width):
    """Scores every pixel in ``im`` by applying ``model`` to windows of ``width`` onto the image"""
    padding = ((width/2, width/2), (width/2, width/2), (0, 0))
    im = sp.pad(im, padding, mode='reflect')    
    
    im = (im - im.mean())/im.std()
    window_centers = get_window_centers(im, width=width)
    window_gen = window_generator(im, window_centers, width=width)
    
    score_list = score_windows(window_gen, model, total_count=len(window_centers))
    unpadded_window_centers = window_centers - width/2    
    
    return unpadded_window_centers, score_list

def score_images(h5file, ims, model, width):
    """
    Scores every pixel in each im of ``ims`` by applying ``model`` to windows of ``width`` about each pixel, and 
    stores the results as arrays in ``h5file``.
    
    Parameters
    ----------
    h5file : h5py.File
        A HDF5 file object that the score arrays will be stored in. The results for image ``file_id`` will be stored at
        ``h5file[file_id]``.
    ims : dict
        A dictionary of ``(file_id, image)`` pairs. 
    model : caffe.Classifier
        A Caffe classifier.
    width :
        The width of the windows to be used for scoring the pixels in each image.
    """
    for i, (file_id, im) in enumerate(ims.items()): 
        logging.info('Processing file {0}, {1} of {2}'.format(file_id, i+1, len(ims)))
        window_centers, score_list = score_image(im, model, width=width)
        score_array = make_score_array(score_list[:, 1], window_centers, im.shape[:2])
        h5file[file_id] = score_array  

def create_classifier(definition_path, model_path):
    """Creates a ``caffe.Classifier`` from the .prototxt at ``definition_path`` and the .modelfile at ``model_path``."""
    caffe.set_mode_gpu()
    m = caffe.Classifier(definition_path, model_path)

    return m