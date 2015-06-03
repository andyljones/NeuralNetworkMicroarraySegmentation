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
This module uses a trained neural network to segment experimental microarray images into foreground and background 
pixels. It then uses the segmentation to measure the expression ratio in each spot.

In detail, ``score_experimental_images`` feeds the experimental images through the NN and stores the calculated 
score array in a HDF5 file. The ``measure_all`` function reads the segmentations out of the HDF5 file and uses them to 
measure the expression ratios. 

``measure_all`` finds the center of each logically-addressed spot in the score array using a simple Hough-lines based 
heuristic. It then assigns each connected chunk of foreground pixels to belong to the spot with the nearest center.
"""
    
import scipy as sp
import skimage
import itertools as it
import scipy.ndimage
import skimage.transform
import h5py

from experimental_tools import get_bounded_im, get_bounded_ims, WINDOW_WIDTH
from caffe_tools import create_classifier, score_images

"""The IDs of the files to be used for testing."""
TEST_IDS = ['3-{0}_pmt100'.format(i) for i in range(13, 17)]

"""The path to the file describing the Caffe classifier."""
DEFINITION_PATH = r'sources/definitions/experimental_deploy.prototxt'

"""The path to the file containing the trained Caffe model"""
MODEL_PATH = 'temporary/models/experimental_iter_24000.caffemodel'

"""The path to the file containing the score arrays for each image"""
SCORES_PATH = 'temporary/scores/experimental_scores.hdf5' 

def score_experimental_images():
    """Scores each pixel in each experimental image using a Caffe classifier, and stores the results in a HDF5 file."""
    classifier = create_classifier(DEFINITION_PATH, MODEL_PATH)
    ims = get_bounded_ims()
    score_images(SCORES_PATH, ims, classifier, WINDOW_WIDTH)

def get_data(file_id):
    """Gets the image and the score array associated with ``file_id``"""
    im, truth = get_bounded_im(file_id)
    with h5py.File(SCORES_PATH, 'r') as h5file:
        score_array = h5file[file_id]        

    return im, score_array
    
def find_grid_angle(array):
    """Finds the angle from horizontal of a microarray segmentation given by ``score_array``."""
    array = (array > 0.5)
    
    theta_ranges = sp.pi/2 + sp.linspace(-0.25, 0.25, 501)
    acc, thetas, rs = skimage.transform.hough_line(array, theta_ranges)
    hspace, angles, dists = skimage.transform.hough_line_peaks(acc, thetas, rs, threshold=200)
    
    most_common_angle = theta_ranges[sp.argmax((acc**2).sum(0))]
    
    return most_common_angle
    
def find_lines_at_angle(array, angle):
    """
    Finds a set of prominent, well-spaced Hough lines in the ``array`` at roughly ``angle`` from horizontal.
    """
    array = (array > 0.5)
    
    threshold = sp.sqrt((sp.sin(angle)*array.shape[1])**2 + (sp.cos(angle)*array.shape[0])**2)/15
    theta_ranges = angle + sp.linspace(-0.001, 0.001, 21)
    acc, thetas, rs = skimage.transform.hough_line(array, theta_ranges)
    hspace, angles, dists = skimage.transform.hough_line_peaks(acc, thetas, rs, threshold=threshold, min_distance=5, min_angle=21)
    
    return angles, dists

def draw_hough_line(array, theta, r, value=1):
    """Replaces all the pixels in ``array`` that fall under a Hough line with parameters ``(theta, r) with ``value``."""
    h, w = array.shape    
    
    x_bounds = [r/sp.cos(theta) - (h - 1)*sp.tan(theta), r/sp.cos(theta)]
    x_min = max(min(x_bounds), 0)
    x_max = min(max(x_bounds), w - 1)
    
    xs = sp.array([x_min, x_max])
    ys = -xs/sp.tan(theta) + r/sp.sin(theta)
  
    line_indices = skimage.draw.line(int(ys[0]), int(x_min), int(ys[1]), int(x_max))
    array[line_indices] = value

def draw_hough_lines(shape, angles, dists):
    """Creates a ``-1``-filled array of ``shape`` and draws the Hough line described by each``(angle, dist)`` pair
    onto the array with value ``1``"""
    array = -sp.ones(shape)
    
    for theta, r in zip(angles, dists):
        draw_hough_line(array, theta, r)
        
    return array

def draw_enumerated_lines(shape, thetas, rs):
    """Creates a ``-1``-filled array of ``shape``, sorts the Hough lines described by ``(thetas, rs)`` in order of 
    ``rs``, then draws them onto the array with the ``i``th line being drawn with value ``i``."""
    array = -sp.ones(shape)
    sorting_idxs = sp.argsort(rs)
    rs = rs[sorting_idxs]
    thetas = thetas[sorting_idxs]
    
    for i, (theta, r) in enumerate(zip(thetas, rs)):
        draw_hough_line(array, theta, r, value=i)
        
    return array

def find_hough_intersections(array):
    """Estimates a set of horizontal and vertical Hough lines from ``array`` and returns two arrays; one containing the
    physical locations of each intersection, and one containing the logical coordinates of each intersection."""
    grid_angle = find_grid_angle(array > 0.5)
    x_thetas, x_rs = find_lines_at_angle(array > 0.5, grid_angle)
    y_thetas, y_rs = find_lines_at_angle(array > 0.5, grid_angle + sp.pi/2)

    enumerated_horizontals = draw_enumerated_lines(array.shape, x_thetas, x_rs)
    enumerated_verticals = draw_enumerated_lines(array.shape, y_thetas, y_rs)
    
    intersections = (enumerated_horizontals != -1) & (enumerated_verticals != -1)
    physical_indices = sp.indices(array.shape)[:, intersections]
    vertical_logical_indices = enumerated_verticals[physical_indices[0], physical_indices[1]]
    horizontal_logical_indices = enumerated_horizontals[physical_indices[0], physical_indices[1]]
    logical_indices = sp.array([horizontal_logical_indices, vertical_logical_indices])
    
    return physical_indices, logical_indices

def find_centroids(array):
    """Finds the centroid of each connected area in ``array``."""
    labelled = sp.ndimage.label(array)[0] - 1
    foreground_indices = sp.indices(labelled.shape)[:, labelled > -1]
    labels_of_indices = labelled[labelled > -1]
    
    # What's happening here is that there are thousands of connected areas in ``array``, and finding the centroid of 
    # each one by manually going ``indices[:, labelled == label]`` is really slow. Instead, we (implicitly) build a 
    # tree. Each node ``n`` in the tree has a label. That node finds the indices corresponding to its own label, then
    # delegates finding the indices corresponding to all smaller labels to its left child, and the indices corresponding
    # to all larger labels to it's right child.
    #
    # Update: I've since learnt about scipy.ndimage.labelled_comprehension. That's a much better way to do this! Leaving
    # this where it is though since well, it works and it's inefficiency is not currently a bottleneck.
    def get_centroids(indices, labels):
        if len(labels) > 0:
            pivot = labels[len(labels)/2]
    
            left_selector = labels < pivot
            left_indices = indices[:, left_selector]
            left_labels = labels[left_selector]
            left_centroids = get_centroids(left_indices, left_labels)
    
            equal_selector = labels == pivot
            equal_indices = indices[:, equal_selector]
            equal_centroid = [equal_indices.mean(1)]        
    
            right_selector = labels > pivot
            right_indices = indices[:, right_selector]
            right_labels = labels[right_selector]
            right_centroids = get_centroids(right_indices, right_labels)
    
            return left_centroids + equal_centroid + right_centroids
        else:
            return []
        
    centroids = sp.array(get_centroids(foreground_indices, labels_of_indices))
    centroids = centroids.astype(int).T
    
    return labelled, centroids
    
def make_index_map(shape, indices):
    """Creates a ``-1``-filled array of ``shape``, then sets the location at ``indices[:, i]`` to ``i``."""
    result = -sp.ones(shape, dtype=int)
    result[indices[0], indices[1]] = range(indices.shape[1])
    return result
    
def shape_containing_indices(*indices):
    """Given one or more lists of indices, finds the smallest shape that could contain them"""
    indices = sp.concatenate(indices, 1)
    maxes = indices.max(1)
    return maxes + 1

def indices_to_boolean_array(indices, shape=None):
    """Creates a zeroed boolean array with every location in ``indices`` set to True"""
    shape = shape_containing_indices(indices) if shape is None else shape
    array = sp.zeros(shape, dtype=bool)
    array[indices[0], indices[1]] = True
    
    return array

def find_too_distant_centroids(intersection_pis, centroid_is, threshold=8):
    """Returns a mask indicating which indices in ``centroid_is`` are more than ``threshold`` distant from an 
    intersection."""
    shape = shape_containing_indices(intersection_pis, centroid_is)
    intersections = indices_to_boolean_array(intersection_pis, shape)
    
    distance_to_intersection_map = sp.ndimage.distance_transform_edt(~intersections)
    distance_to_intersection = distance_to_intersection_map[centroid_is[0], centroid_is[1]]
    
    too_distant_centroids = sp.arange(centroid_is.shape[1])[distance_to_intersection > threshold]
    
    return too_distant_centroids
    
def assign_centroids_to_intersections(intersection_pis, centroid_is):
    """Returns an array where the ``i``th entry is the index of the closest intersection in ``intersection_pis`` to the
    ``i`th centroid in ``centroid_is``."""
    shape = shape_containing_indices(intersection_pis, centroid_is)
    intersections = indices_to_boolean_array(intersection_pis, shape)
    
    nearest_intersection_map = sp.ndimage.distance_transform_edt(~intersections, return_distances=False, return_indices=True)
    indices_of_nearest_intersection = nearest_intersection_map[:, centroid_is[0], centroid_is[1]]
    
    intersection_index_map = make_index_map(shape, intersection_pis)
    number_of_nearest_intersection = intersection_index_map[indices_of_nearest_intersection[0], indices_of_nearest_intersection[1]]
    
    return number_of_nearest_intersection

def expand_labels(labelled, array):
    """Replaces each pixel >0.5 in ``array`` with the label of the nearest non-zero pixel in ``labelled``."""
    nearest_label_indices = sp.ndimage.distance_transform_edt((labelled == -1), return_distances=False, return_indices=True)
    nearest_label = labelled[nearest_label_indices[0], nearest_label_indices[1]]
    
    return (array > 0.5)*nearest_label

def suppress_labels(labels, labelled):
    """Sets each pixel in a copy of ``labelled`` with a label from ``labels`` to -1"""
    labelled = sp.copy(labelled)
    to_suppress = sp.in1d(labelled, labels).reshape(labelled.shape)
    labelled[to_suppress] = -1
    return labelled

def find_multi_centroids(nearest_intersection):
    """Finds the centroids which map non-injectively onto the nearest intersection."""
    numbers = [[] for _ in range(max(nearest_intersection) + 1)]
    for i in range(len(nearest_intersection)):
        numbers[nearest_intersection[i]].append(i)
        
    multiple = [n for n in numbers if len(n) > 1]
    
    return multiple
    
def make_labelling_consistent(labelling, nearest_intersections):
    """Removes any label from a copy of ``labelling`` that is stopping ``nearest_intersections`` from being an 
    injective map from labels onto intersections."""
    labelling = sp.copy(labelling)
    multiple = find_multi_centroids(nearest_intersections)
    for labels in multiple:
        label_to_use = labels[0] #TODO: Use the largest label rather than just the first
        for label in labels:
            labelling[labelling == label] = label_to_use
            
    return labelling

def label_by_logical_index(array):
    """Labels a score array with a logical indexing. If a pixel belongs to the (i,j)th spot in the grid, then the label
    of that spot in ``consistent_labelling`` will map to ``(i, j)`` in ``nearest_intersection_lis``."""
    labelling, centroid_pis = find_centroids(array > 0.5)    
    intersection_pis, intersection_lis = find_hough_intersections(array)
    
    too_distant_labels = find_too_distant_centroids(intersection_pis, centroid_pis)
    good_labelling = suppress_labels(too_distant_labels, labelling)
    nearest_intersections = assign_centroids_to_intersections(intersection_pis, centroid_pis)
    nearest_intersection_lis = intersection_lis[:, nearest_intersections]
    
    consistent_labelling = make_labelling_consistent(good_labelling, nearest_intersections)
    
    return consistent_labelling, nearest_intersection_lis
    
def measure_labelling(im, labelling, lis):
    """Given a one-channel image, a labelling and a map of labels onto logical indices, returns an array
    with the median value of the part of ``im`` labelled ``l`` stored at location ``lis[l]``."""
    labels = sp.unique(labelling)
    labels = labels[labels != -1]    
    
    measurements = sp.ndimage.labeled_comprehension(im, labelling, labels, sp.median, float, sp.nan)

    results = sp.zeros((64, 192))
    results[:, :] = sp.nan
    restricted_lis = lis[:, labels].astype(int)
    results[restricted_lis[0], restricted_lis[1]] = measurements
        
    return results

def morphological_background(im):
    """Estimates the background at each pixel by taking the minimum over a 50-pixel-radius disk."""
    opened = sp.ndimage.grey_opening(im, 50)
    return opened
    
def measure(im, array):
    """Calculates the morphologically-corrected expression ratio for each spot in ``im`` using a segmentation provided
    by ``array``."""
    labelling, lis = label_by_logical_index(array)
    
    measures = []
    for channel in im:
        spot_measures = measure_labelling(channel, labelling, lis)
        background = morphological_background(channel)
        background_measures = measure_labelling(background, labelling, lis) 
        measures.append(sp.exp(spot_measures) - sp.exp(background_measures)) # The images are read in in logspace
        
    return measures[0]/measures[1]

def measure_all():
    """Calculates expression ratios for each spot in each experimental image, using segmentations from the HDF5 file."""
    results = []
    for im_id in TEST_IDS:
        im, array = get_data(im_id)
        result = measure(im, array)
        results.append(result)
        
    valid_locations = ~reduce(sp.bitwise_or, map(sp.isnan, results))
    valid_locations[valid_locations == True] = sp.nan
    valid_results = sp.array([r*valid_locations for r in results])
    
    return valid_results
    
def correlations(results):
    """Calculates the interreplicate pairwise correlations among measurements generated by ``measure_all``"""
    valid_locations = ~sp.isnan(results.sum(0)) & sp.isfinite(results.sum(0))
    flat_results = results[:, valid_locations].reshape((4, -1))
    corrs = sp.corrcoef(flat_results)
    return corrs
    
def mean_absolute_errors(results):
    """Calculates the interreplicate pairwise mean absolute errors among measurements generated by ``measure_all``"""
    valid_locations = ~sp.isnan(results.sum(0)) & sp.isfinite(results.sum(0))
    flat_results = results[:, valid_locations].reshape((4, -1))
    
    maes = []
    for r1, r2 in it.combinations(flat_results, 2):
        maes.append(sp.mean(sp.absolute(r1 - r2)))

    return maes   
