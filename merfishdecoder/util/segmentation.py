import numpy as np
import geopandas as geo
import pandas as pd
from skimage.segmentation import find_boundaries
from shapely.geometry import Polygon
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from merfishdecoder.core import zplane
from merfishdecoder.core import dataset

from cellpose import utils
from cellpose import models

def connect_features_per_fov(
    dataSet,
    features: geo.geodataframe.GeoDataFrame,
    bufferSize = 15,
    fov: int = None):
    
    x = features[features.fov == fov]
    
    graph = np.zeros((
        x.shape[0], 
        x.shape[0]))
    for i in range(x.shape[0]):
        for j in range(x.shape[0]):
            p1 = x.geometry.iloc[i].centroid.buffer(bufferSize)
            p2 = x.geometry.iloc[j].centroid.buffer(bufferSize)
            if p1.intersects(p2):
                graph[i, j] = 1
    graph = csr_matrix(graph)
    n_components, labels = connected_components(
        csgraph=graph, directed=False, return_labels=True)
    return x.assign(
        name= ["fov_%d_feature_%d" % (fov, x) for x in labels])


def extract_polygon_per_index(
    img, idx):
    
    """Extract features from a segmented image. 
 
    Parameters
    ----------
    img : np.array
        Segmented image.

    idx : int
        Index of the feature. 

    Returns
    -------
    A Polygon object. 
    """
    
    from functools import reduce
    import operator
    import math
    
    (y, x) = np.where(
        find_boundaries(
        img == idx, mode='inner'))
    points = np.array([x, y]).T

    if points.shape[0] == 0:
        return None
    else:
        hull = None
        if (points[:,0].max() - points[:,0].min() > 0) & \
            (points[:,1].max() - points[:,1].min() > 0):
            coords = [[x, y] for (x, y) in points]
            center = tuple(map(operator.truediv, reduce(lambda x, y: map(operator.add, x, y), coords), [len(coords)] * 2))
            pointsOrdered = sorted(coords, key=lambda coord: (-135 - math.degrees(math.atan2(*tuple(map(operator.sub, coord, center))[::-1]))) % 360)
            hull = Polygon(pointsOrdered)
        return hull

def run_cell_pose(
    gpu = False,
    modelType = "nuclei",
    images: list = None,
    diameter: int = 150,
    channels: list = None,
    do_3D: bool = False
    ) -> np.ndarray:
    
    """Run cell pose for cell segmentation
 
    Parameters
    ----------
    gpu : bool
        A boolen variable indicates whether to use GPU
    model_type : str
        Type of segmentation (nuclei or cyto)
    images : np.ndarray
        Input image stack for segmentation.
    diameter : int
        Average diameter for features
    channels : list
        list of channels, either of length 2 or of length number of images by 2.
        First element of list is the channel to segment (0=grayscale, 1=red, 2=blue, 3=green).
        Second element of list is the optional nuclear channel (0=none, 1=red, 2=blue, 3=green).
        For instance, to segment grayscale images, input [0,0]. To segment images with cells
        in green and nuclei in blue, input [2,3]. To segment one grayscale image and one
        image with cells in green and nuclei in blue, input [[0,0], [2,3]].
    do_3D: bool
        set to True to run 3D segmentation on 4D image input

    Returns
    -------
    masks: list of 2D arrays, or single 3D array (if do_3D=True)
        labelled image, where 0=no masks; 1,2,...=mask labels
    """
    
    model = models.Cellpose(
        gpu = gpu, 
        model_type = modelType)
    
    masks, flows, styles, diams = \
        model.eval(
        images, 
        diameter = diameter,
        do_3D = do_3D, 
        channels= [[0, 0]] * len(images))

    return masks

def global_align_features_per_fov(
    dataSet,
    features: geo.geodataframe.GeoDataFrame,
    fov: int = None):
    x = features[features.fov == fov]
    x = x.assign(geometry = x.affine_transform(
        [dataSet.get_microns_per_pixel(), 0, 
        0, dataSet.get_microns_per_pixel(), 
        dataSet.get_fov_offset(fov)[0], 
        dataSet.get_fov_offset(fov)[1]]))
    
    x = x.assign(global_x = x.centroid.x)
    x = x.assign(global_y = x.centroid.y)
    return x

def filter_features_per_fov(
    dataSet,
    features: geo.geodataframe.GeoDataFrame,
    fov: int = None,
    minZplane: int = 2,
    borderSize: int = 80):
    
    def check_duplicate(x):
        return len(x) != len(set(x))

    x = features[features.fov == fov]
    
    centroids = pd.DataFrame([[ 
        x[x.name == i].x.mean(),
        x[x.name == i].y.mean(),
        x[x.name == i].shape[0],
        i, 
        check_duplicate(x[x.name == i].z)] \
        for i in np.unique(x.name) ],
        columns = ["x", "y", "z", "name", "duplicate"])
    
    xWidth = dataSet.get_image_dimensions()[0]
    yWidth = dataSet.get_image_dimensions()[1]
    
    centroids = centroids.assign(edge = \
        ( centroids.x < borderSize ) | \
        ( centroids.x > xWidth - borderSize ) | \
        ( centroids.y < borderSize ) | \
        ( centroids.y > yWidth - borderSize ))
    
    centroids = centroids[
        (centroids.z >= minZplane) & \
        (centroids.z <= len(dataSet.get_z_positions())) & \
        ~centroids.edge & \
        ~centroids.duplicate ]
    
    return x[x.name.isin(centroids.name)]
