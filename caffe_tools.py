# -*- coding: utf-8 -*-
"""
Created on Sat May 30 12:16:03 2015

@author: andy
"""

import scipy as sp
import caffe
import lmdb
import os
import datetime
import itertools as it

def get_window(im, center, width):
    if width % 2 == 0:
        i_slice = slice(center[0]-width/2, center[0]+width/2)
        j_slice = slice(center[1]-width/2, center[1]+width/2)
    else:
        i_slice = slice(center[0]-width/2, center[0]+width/2+1)
        j_slice = slice(center[1]-width/2, center[1]+width/2+1)
        
    return im[i_slice, j_slice]

def randomly_flip(im):
    if sp.rand() > 0.5:
        im = im[::-1, :]
    if sp.rand() > 0.5:
        im = im[:, ::-1]
        
    return im    
    
def randomly_rotate(im):
    k = sp.random.randint(0, 4)
    return sp.ndimage.interpolation.rotate(im, 90*k)
    
def make_datum(im, center, label, width):
    window = get_window(im, center, width)
    window = randomly_flip(window)
    window = randomly_rotate(window)
    window = sp.rollaxis(window, 2, 0)
    datum = caffe.io.array_to_datum(window.astype(float), label)
    return datum

def progress_report(count_so_far, total_count, start_time):
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
    start_time = datetime.datetime.now()
    
    if os.path.exists(name): raise ValueError('Database {0} already exists'.format(name))     
    
    env = lmdb.open(name, map_size=1e12)
    with env.begin(write=True) as txn:
        for i, (center, label) in enumerate(zip(centers.T, labels)):
            im_no = center[2]
            datum = make_datum(ims[im_no], center[:2], label, width)
            key = '{0}_{1}-{2}-{3}'.format(i, im_no, center[0], center[1])
            txn.put(key, datum.SerializeToString())
            
            if i % 500 == 0: print(progress_report(i+1, centers.shape[1], start_time))
                
    env.close()
            
def load_from_db(name, key):
    env = lmdb.open(name)
    with env.begin(write=False) as txn:
        s = txn.get(key)
        datum = caffe.proto.caffe_pb2.Datum()
        datum.ParseFromString(s)
        
        arr = caffe.io.datum_to_array(datum)
        
        return arr, datum.label

def make_score_array(score_list, window_centers, shape):
    score_array = sp.zeros(shape)
    score_array[window_centers[:, 0], window_centers[:, 1]] = score_list

    return score_array
    
def get_window_centers(im, k, width=64):
    boundary = int(sp.ceil(width/2))
    yslice = slice(boundary, im.shape[0]-boundary, k)
    xslice = slice(boundary, im.shape[1]-boundary, k)
    indices = sp.mgrid[yslice, xslice]
    indices = sp.rollaxis(indices, 0, 3).reshape((-1, 2))
    
    return indices

def window_generator(im, window_centers, width=64):
    for window_center in window_centers:
        window = get_window(im, window_center, width)
        yield window
        
    return 

def score_window_list(window_list, model):
    unflipped = sp.array(window_list)
    flipped_lr = unflipped[:, :, ::-1]
    flipped_ud = unflipped[:, ::-1, :]
    flipped_lrud = unflipped[:, ::-1, ::-1]

    flipped = [unflipped, flipped_lr, flipped_ud, flipped_lrud]

    rotated = [sp.ndimage.interpolation.rotate(f, 90*k, axes=(1, 2), order=0) for f in flipped for k in range(3)]

    predictions = [model.predict(ws, oversample=False) for ws in rotated] 
    return sp.median(sp.array(predictions), 0)

def score_windows(window_generator, model, total_count=0):
    start_time = datetime.datetime.now()
    count_so_far = 0    
    
    results = []    
    while True:
        windows = list(it.islice(window_generator, 4096))
#        pdb.set_trace()
        if not windows:
            break
        
        result = score_window_list(windows, model)
        results.append(result)
        
        count_so_far = count_so_far + len(result)
        
        if total_count:
            print(progress_report(count_so_far, total_count, start_time))
        else:
            print('Processed {0} window_centers'.format(count_so_far))

        
    results = sp.concatenate(results)
    
    return results
    
def masked_scores(shape, window_centers, scores):
    mask = sp.ones(shape, dtype=bool)
    mask[window_centers[:, 0], window_centers[:, 1]] = False    
    
    data = sp.zeros(shape)
    data[window_centers[:, 0], window_centers[:, 1]] = scores
    masked_scores = sp.ma.MaskedArray(data, mask=mask)
    
    return masked_scores
    
def score_image(im, model, k=1, width=64):
    padding = ((width/2, width/2), (width/2, width/2), (0, 0))
    im = sp.pad(im, padding, mode='reflect')    
    
    im = (im - im.mean())/im.std()
    window_centers = get_window_centers(im, k, width=width)
    window_gen = window_generator(im, window_centers, width=width)
    
    score_list = score_windows(window_gen, model, total_count=len(window_centers))
    
    return window_centers, score_list

def create_classifier(model_file, pretrained):
    caffe.set_mode_gpu()
    m = caffe.Classifier(model_file, pretrained)

    return m