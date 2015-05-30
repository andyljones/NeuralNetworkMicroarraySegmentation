# -*- coding: utf-8 -*-
"""
Created on Sat May 30 12:19:49 2015

@author: andy
"""

    
import scipy as sp
#import caffe
import experimental_io as eio
import caffe_tools as ct
import skimage
import itertools as it
import scipy.ndimage
import skimage.transform
import cPickle
import seaborn as sns
import hand_label_extractor 
import graphing_tools


def get_data(file_id, channel=0):
    im = get_bounded_im(file_id, channel=channel)
    window_centers = sp.load('temporary/scores_0/{0}_window_centers.npy'.format(file_id))
    score_list = sp.load('temporary/scores_0/{0}_score_list.npy'.format(file_id))

    return im, window_centers - 32, score_list

example_file_id = '3-12_pmt100'

test_ids = ['3-{0}_pmt100'.format(i) for i in range(13, 17)]

MODEL_FILE = r'caffe_definitions/experimental_deploy.prototxt'
PRETRAINED = 'caffe_definitions/experimental_2_384k.caffemodel'

sns.set_context({'figure.figsize': (18,18)})

def score_images():
    classifier = ct.create_classifier(MODEL_FILE, PRETRAINED)
    ims = eio.get_bounded_ims()
    for i, (file_id, im) in enumerate(ims.items()): 
        print('Processing file {0}, {1} of {2}'.format(file_id, i+1, len(ims)))
        window_centers, score_list = ct.score_image(im, classifier)
        sp.save('temporary/scores/' + file_id + '_score_list', score_list)
        sp.save('temporary/scores/' + file_id + '_window_centers', window_centers)    


def load_score_array(file_id, channel=0):
    im, centers, scores = eio.get_data(file_id, channel=-1)
    score_array = ct.make_score_array(1 - scores[:, 0], centers, im.shape[1:])
    return im, score_array
    
def find_grid_angle(array):
    array = (array > 0.5)
    
    theta_ranges = sp.pi/2 + sp.linspace(-0.25, 0.25, 501)
    acc, thetas, rs = skimage.transform.hough_line(array, theta_ranges)
    hspace, angles, dists = skimage.transform.hough_line_peaks(acc, thetas, rs, threshold=200)
    
    most_common_angle = theta_ranges[sp.argmax((acc**2).sum(0))]
    
    return most_common_angle
    
def find_lines_at_angle(array, angle):
    array = (array > 0.5)
    
    threshold = sp.sqrt((sp.sin(angle)*array.shape[1])**2 + (sp.cos(angle)*array.shape[0])**2)/15
    theta_ranges = angle + sp.linspace(-0.001, 0.001, 21)
    acc, thetas, rs = skimage.transform.hough_line(array, theta_ranges)
    hspace, angles, dists = skimage.transform.hough_line_peaks(acc, thetas, rs, threshold=threshold, min_distance=5, min_angle=21)
    
    return angles, dists
    
def show_hough_lines(im, angles, dists):
    fig, ax = ct.show_image(im)
    for angle, dist in zip(angles, dists):
        xs = sp.array([0, im.shape[1]])
        ys = -xs/sp.tan(angle) + dist/sp.sin(angle)
    
        ax.plot(xs, ys, 'r')
    
    ax.set_xlim(0, im.shape[1])
    ax.set_ylim(im.shape[0], 0)
    
    return fig, ax    

def draw_hough_line(array, theta, r, value=1):
    h, w = array.shape    
    
    x_bounds = [r/sp.cos(theta) - (h - 1)*sp.tan(theta), r/sp.cos(theta)]
    x_min = max(min(x_bounds), 0)
    x_max = min(max(x_bounds), w - 1)
    
    xs = sp.array([x_min, x_max])
    ys = -xs/sp.tan(theta) + r/sp.sin(theta)
  
    line_indices = skimage.draw.line(int(ys[0]), int(x_min), int(ys[1]), int(x_max))
    array[line_indices] = value

def draw_hough_lines(shape, angles, dists):
    array = -sp.ones(shape)
    
    for theta, r in zip(angles, dists):
        draw_hough_line(array, theta, r)
        
    return array

def draw_enumerated_lines(shape, thetas, rs):
    array = -sp.ones(shape)
    sorting_idxs = sp.argsort(rs)
    rs = rs[sorting_idxs]
    thetas = thetas[sorting_idxs]
    
    for i, (theta, r) in enumerate(zip(thetas, rs)):
        draw_hough_line(array, theta, r, value=i)
        
    return array

def find_hough_intersections(array):
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
    labelled = sp.ndimage.label(array)[0] - 1
    foreground_indices = sp.indices(labelled.shape)[:, labelled > -1]
    labels_of_indices = labelled[labelled > -1]
    
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
    result = -sp.ones(shape, dtype=int)
    result[indices[0], indices[1]] = range(indices.shape[1])
    return result
    
def shape_containing_indices(*indices):
    indices = sp.concatenate(indices, 1)
    maxes = indices.max(1)
    return maxes + 1

def indices_to_boolean_array(indices, shape=None):
    shape = shape_containing_indices(indices) if shape is None else shape
    array = sp.zeros(shape, dtype=bool)
    array[indices[0], indices[1]] = True
    
    return array

def find_too_distant_centroids(intersection_pis, centroid_is, threshold=8):
    shape = shape_containing_indices(intersection_pis, centroid_is)
    intersections = indices_to_boolean_array(intersection_pis, shape)
    
    distance_to_intersection_map = sp.ndimage.distance_transform_edt(~intersections)
    distance_to_intersection = distance_to_intersection_map[centroid_is[0], centroid_is[1]]
    
    too_distant_centroids = sp.arange(centroid_is.shape[1])[distance_to_intersection > threshold]
    
    return too_distant_centroids
    
def assign_centroids_to_intersections(intersection_pis, centroid_is):
    shape = shape_containing_indices(intersection_pis, centroid_is)
    intersections = indices_to_boolean_array(intersection_pis, shape)
    
    nearest_intersection_map = sp.ndimage.distance_transform_edt(~intersections, return_distances=False, return_indices=True)
    indices_of_nearest_intersection = nearest_intersection_map[:, centroid_is[0], centroid_is[1]]
    
    intersection_index_map = make_index_map(shape, intersection_pis)
    number_of_nearest_intersection = intersection_index_map[indices_of_nearest_intersection[0], indices_of_nearest_intersection[1]]
    
    return number_of_nearest_intersection

def expand_labels(labelled, array):
    nearest_label_indices = sp.ndimage.distance_transform_edt((labelled == -1), return_distances=False, return_indices=True)
    nearest_label = labelled[nearest_label_indices[0], nearest_label_indices[1]]
    
    return (array > 0.5)*nearest_label

def suppress_labels(labels, labelled):
    labelled = sp.copy(labelled)
    to_suppress = sp.in1d(labelled, labels).reshape(labelled.shape)
    labelled[to_suppress] = -1
    return labelled

def find_multi_centroids(nearest_intersection):
    numbers = [[] for _ in range(max(nearest_intersection) + 1)]
    for i in range(len(nearest_intersection)):
        numbers[nearest_intersection[i]].append(i)
        
    multiple = [n for n in numbers if len(n) > 1]
    
    return multiple
    
def make_labelling_consistent(labelling, nearest_intersections):
    labelling = sp.copy(labelling)
    multiple = find_multi_centroids(nearest_intersections)
    for labels in multiple:
        label_to_use = labels[0]
        for label in labels:
            labelling[labelling == label] = label_to_use
            
    return labelling

def label_by_logical_index(array):
    labelling, centroid_pis = find_centroids(array > 0.5)    
    intersection_pis, intersection_lis = find_hough_intersections(array)
    
    too_distant_labels = find_too_distant_centroids(intersection_pis, centroid_pis)
    good_labelling = suppress_labels(too_distant_labels, labelling)
    nearest_intersections = assign_centroids_to_intersections(intersection_pis, centroid_pis)
    nearest_intersection_lis = intersection_lis[:, nearest_intersections]
    
    consistent_labelling = make_labelling_consistent(good_labelling, nearest_intersections)
    
    return consistent_labelling, nearest_intersection_lis
    
def measure_labelling(im, labelling, lis):
    labels = sp.unique(labelling)
    labels = labels[labels != -1]    
    
    measurements = sp.ndimage.labeled_comprehension(im, labelling, labels, sp.median, float, sp.nan)

    results = sp.zeros((64, 192))
    results[:, :] = sp.nan
    restricted_lis = lis[:, labels].astype(int)
    results[restricted_lis[0], restricted_lis[1]] = measurements
        
    return results

def morphological_background(im):
    opened = sp.ndimage.grey_opening(im, 50)
    return opened
    
def measure(im, array):
    labelling, lis = label_by_logical_index(array)
    
    measures = []
    for channel in im:
        spot_measures = measure_labelling(channel, labelling, lis)
        background = morphological_background(channel)
        background_measures = measure_labelling(background, labelling, lis) 
        measures.append(sp.exp(spot_measures) - sp.exp(background_measures))
        
    return measures[0]/measures[1]

def measure_all(**kwargs):
    results = []
    for im_id in test_ids:
        im, array = load_score_array(im_id, channel=-1)
        result = measure(im, array, **kwargs)
        results.append(result)
        
    valid_locations = ~reduce(sp.bitwise_or, map(sp.isnan, results))
    valid_locations[valid_locations == True] = sp.nan
    valid_results = sp.array([r*valid_locations for r in results])
    
    return valid_results
    
def correlations(results):
    valid_locations = ~sp.isnan(results.sum(0)) & sp.isfinite(results.sum(0))
    flat_results = results[:, valid_locations].reshape((4, -1))
    corrs = sp.corrcoef(flat_results)
    return corrs
    
def mean_absolute_errors(results):
    valid_locations = ~sp.isnan(results.sum(0)) & sp.isfinite(results.sum(0))
    flat_results = results[:, valid_locations].reshape((4, -1))
    
    maes = []
    for r1, r2 in it.combinations(flat_results, 2):
        maes.append(sp.mean(sp.absolute(r1 - r2)))

    return maes   

def save_results():
    results = measure_all()
    corrs = correlations(results)
    maes = mean_absolute_errors(results)
    
    with open('results/experimental_results.pickle', 'w+') as f:
        cPickle.dump({'correlations': corrs, 'mean_absolute_errors': maes}, f)
