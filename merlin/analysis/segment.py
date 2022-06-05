import cv2
import numpy as np
from skimage import measure
from skimage import segmentation
import rtree
import geopandas
from shapely import geometry
from typing import List, Dict
from scipy.spatial import cKDTree

from merlin.core import dataset
from merlin.core import analysistask
from merlin.util import spatialfeature
from merlin.util import watershed
from merlin.util import imagefilters

import pandas
import networkx as nx
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from cellpose import models
import cellpose
from shapely.geometry import Point, LineString, Polygon


def flatten_list(_2d_list):
    flat_list = []
    # Iterate through the outer list
    for element in _2d_list:
        if type(element) is list:
            # If the element is of type list, iterate through the sublist
            for item in element:
                flat_list.append(item)
        else:
            flat_list.append(element)
    return flat_list
    
class FeatureSavingAnalysisTask(analysistask.ParallelAnalysisTask):

    """
    An abstract analysis class that saves features into a spatial feature
    database.
    """

    def __init__(self, dataSet: dataset.DataSet, parameters=None,
                 analysisName=None):
        super().__init__(dataSet, parameters, analysisName)

    def _reset_analysis(self, fragmentIndex: int = None) -> None:
        super()._reset_analysis(fragmentIndex)
        self.get_feature_database().empty_database(fragmentIndex)

    def get_feature_database(self) -> spatialfeature.SpatialFeatureDB:
        """ Get the spatial feature database this analysis task saves
        features into.

        Returns: The spatial feature database reference.
        """
        return spatialfeature.HDF5SpatialFeatureDB(self.dataSet, self)


class WatershedSegment(FeatureSavingAnalysisTask):

    """
    An analysis task that determines the boundaries of features in the
    image data in each field of view using a watershed algorithm.
    
    Since each field of view is analyzed individually, the segmentation results
    should be cleaned in order to merge cells that cross the field of
    view boundary.
    """

    def __init__(self, dataSet, parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)

        if 'seed_channel_name' not in self.parameters:
            self.parameters['seed_channel_name'] = 'DAPI'
        if 'watershed_channel_name' not in self.parameters:
            self.parameters['watershed_channel_name'] = 'polyT'

    def fragment_count(self):
        return len(self.dataSet.get_fovs())

    def get_estimated_memory(self):
        # TODO - refine estimate
        return 2048

    def get_estimated_time(self):
        # TODO - refine estimate
        return 5

    def get_dependencies(self):
        return [self.parameters['warp_task'],
                self.parameters['global_align_task']]

    def get_cell_boundaries(self) -> List[spatialfeature.SpatialFeature]:
        featureDB = self.get_feature_database()
        return featureDB.read_features()

    def _run_analysis(self, fragmentIndex):
        globalTask = self.dataSet.load_analysis_task(
                self.parameters['global_align_task'])

        seedIndex = self.dataSet.get_data_organization().get_data_channel_index(
            self.parameters['seed_channel_name'])
        seedImages = self._read_and_filter_image_stack(fragmentIndex,
                                                       seedIndex, 5)

        watershedIndex = self.dataSet.get_data_organization() \
            .get_data_channel_index(self.parameters['watershed_channel_name'])
        watershedImages = self._read_and_filter_image_stack(fragmentIndex,
                                                            watershedIndex, 5)
        seeds = watershed.separate_merged_seeds(
            watershed.extract_seeds(seedImages))
        
        normalizedWatershed, watershedMask = watershed.prepare_watershed_images(
            watershedImages)

        seeds[np.invert(watershedMask)] = 0
        watershedOutput = segmentation.watershed(
            normalizedWatershed, measure.label(seeds), mask=watershedMask,
            connectivity=np.ones((3, 3, 3)), watershed_line=True)

        zPos = np.array(self.dataSet.get_data_organization().get_z_positions())
        feature_from_label_matrix = [spatialfeature.SpatialFeature.feature_from_label_matrix(
            (watershedOutput == i), fragmentIndex,
            globalTask.fov_to_global_transform(fragmentIndex), zPos)
            for i in np.unique(watershedOutput) if i != 0]

        featureDB = self.get_feature_database()
        featureDB.write_features(featureList, fragmentIndex)

    def _read_and_filter_image_stack(self, fov: int, channelIndex: int,
                                     filterSigma: float) -> np.ndarray:
        filterSize = int(2*np.ceil(2*filterSigma)+1)
        warpTask = self.dataSet.load_analysis_task(
            self.parameters['warp_task'])
        return np.array([cv2.GaussianBlur(
            warpTask.get_aligned_image(fov, channelIndex, z),
            (filterSize, filterSize), filterSigma)
            for z in range(len(self.dataSet.get_z_positions()))])

class CellPoseSegment(FeatureSavingAnalysisTask):

    """
    An analysis task that determines the boundaries of features in the
    image data in each field of view using a cellpose algorithm.
    
    Since each field of view is analyzed individually, the segmentation results
    should be cleaned in order to merge cells that cross the field of
    view boundary.
    """

    def __init__(self, dataSet, parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)

        if 'nuclei_channel_name' not in self.parameters:
            self.parameters['nuclei_channel_name'] = 'DAPI'
        if 'cyto_channel_name' not in self.parameters:
            self.parameters['cyto_channel_name'] = 'polyT'
        if 'model_type' not in self.parameters:
            self.parameters['model_type'] = "cyto2"
        if 'diameter' not in self.parameters:
            self.parameters['diameter'] = 100
        if 'channel1' not in self.parameters:
            self.parameters['channel1'] = 2
        if 'channel2' not in self.parameters:
            self.parameters['channel2'] = 3
        if 'max_clip' not in self.parameters:
            self.parameters['max_clip'] = 0.97
        if 'low_pass_sigma' not in self.parameters:
            self.parameters['low_pass_sigma'] = 1
        if 'flow_threshold' not in self.parameters:
            self.parameters['flow_threshold'] = 0.0
        if 'min_size' not in self.parameters:
            self.parameters['min_size'] = 0
        if 'connect_distance' not in self.parameters:
            self.parameters['connect_distance'] = 5
        if 'n_neighbors' not in self.parameters:
            self.parameters['n_neighbors'] = 6
        if 'resample' not in self.parameters:
            self.parameters['resample'] = True
        if 'normalize' not in self.parameters:
            self.parameters['normalize'] = False
        if 'write_mask_image' not in self.parameters:
            self.parameters['write_mask_images'] = True
        if 'use_gpu' not in self.parameters:
            self.parameters['use_gpu'] = False
        if 'maximum_projection' not in self.parameters:
            self.parameters['maximum_projection'] = False

    def fragment_count(self):
        return len(self.dataSet.get_fovs())

    def get_estimated_memory(self):
        # TODO - refine estimate
        return 2048

    def get_estimated_time(self):
        # TODO - refine estimate
        return 5

    def get_dependencies(self):
        return [self.parameters['warp_task'],
                self.parameters['global_align_task']]

    def get_cell_boundaries(self) -> List[spatialfeature.SpatialFeature]:
        featureDB = self.get_feature_database()
        return featureDB.read_features()

    def _read_image_stack(self, fov: int, channelIndex: int) -> np.ndarray:
        warpTask = self.dataSet.load_analysis_task(
            self.parameters['warp_task'])
        return np.array([warpTask.get_aligned_image(fov, channelIndex, z)
                         for z in range(len(self.dataSet.get_z_positions()))])

    def _connect_2D_masks_to_3D_masks(self, 
                                     masks2D,
                                     fragmentIndex,
                                     n_neighbors,
                                     distance_cutoff: float = 3.0):
        
        from sklearn.neighbors import NearestNeighbors
        import numpy as np
        
        globalTask = self.dataSet.load_analysis_task(
            self.parameters['global_align_task'])
        zPos = self.dataSet.get_data_organization().get_z_positions()
        features = []
        for i in range(masks2D.shape[0]):
            masks3D = np.zeros(masks2D.shape)
            masks3D[i,:,:] = masks2D[i,:,:]
            # obtain the z index for each feature
            features_z = [ (i, label, spatialfeature.SpatialFeature.feature_from_label_matrix(
                    masks3D == label, fragmentIndex, 
                    globalTask.fov_to_global_transform(fragmentIndex),
                    list(np.unique(np.where(masks3D == label)[0])))) \
                    for label in np.unique(masks3D) if label != 0 ]
            features.extend(features_z)
        
        # copy the feature list
        zIndexList = [ z for z, old_label, ft in features if ft != None ]
        oldLabelList = [ old_label for z, old_label, ft in features if ft != None]
        featuresList = [ ft for z, old_label, ft in features if ft != None]
        
        # if there is a single features
        if len(featuresList) <= 1:
            return masks2D.astype(np.uint16)

        # get the centroid positions
        centroids = np.array([[
            (x.get_bounding_box()[0] + x.get_bounding_box()[2]) / 2,
            (x.get_bounding_box()[1] + x.get_bounding_box()[3]) / 2,
             zPos[x.get_z_coordinates()[0]]] \
            for x in featuresList ])
        
        # find k nearest neighbours
        n_neighbors = min(n_neighbors, centroids.shape[0])
        nbrs = NearestNeighbors(n_neighbors=n_neighbors, 
                                algorithm='ball_tree').fit(centroids)
        distances, indices = nbrs.kneighbors(centroids)
        
        # create knn graph
        graph = np.zeros((len(featuresList), len(featuresList)))
        np.fill_diagonal(graph, 1)
        for i in range(len(indices)):
            idx = indices[i]
            dst = distances[i]
            zpos_i = featuresList[idx[0]].get_z_coordinates()[0]
            for ii, dd in zip(idx, dst):
                zpos_j = featuresList[ii].get_z_coordinates()[0]
                if dd < distance_cutoff and zpos_i != zpos_j:
                    graph[idx[0],ii] = 1

        n_components, newLabelList = connected_components(
            csgraph=csr_matrix(graph), directed=False, 
            return_labels=True)
        
        # update the mask image
        masks3D = np.zeros(masks2D.shape)
        for z, o, n in zip(zIndexList, oldLabelList, newLabelList):
            masks3D[z,masks2D[z,] == o] = n
        
        return masks3D.astype(np.uint16)
        
    def _run_analysis(self, fragmentIndex):
        globalTask = self.dataSet.load_analysis_task(
                self.parameters['global_align_task'])

        # read membrane and nuclear indices
        dapi_ids = self.dataSet.get_data_organization().get_data_channel_index(
                self.parameters['nuclei_channel_name'])
        cyto_ids = self.dataSet.get_data_organization().get_data_channel_index(
                self.parameters['cyto_channel_name'])

        # read images and perform segmentation
        dapi_images = self._read_image_stack(fragmentIndex, dapi_ids)
        cyto_images = self._read_image_stack(fragmentIndex, cyto_ids)
		
		# if max projection is true, cell pose will only run on the max projected images
		# this could be improved if cellposes can run on a single image
        if self.parameters['maximum_projection']:
            dapi_images = np.array([ np.amax(dapi_images, axis=0) for i in range(dapi_images.shape[0]) ]) 
            cyto_images = np.array([ np.amax(cyto_images, axis=0) for i in range(cyto_images.shape[0]) ])
        
        # apply low pass filter 
        dapi_images = np.array([ imagefilters.low_pass_filter(
                x, self.parameters['low_pass_sigma']) \
            for x in dapi_images ])
        cyto_images = np.array([ imagefilters.low_pass_filter(
                x, self.parameters['low_pass_sigma']) \
            for x in cyto_images ])

        dapi_images[dapi_images > np.quantile(dapi_images, 
                                              self.parameters['max_clip'])] = \
                np.quantile(dapi_images, self.parameters['max_clip'])
        
        cyto_images[cyto_images > np.quantile(cyto_images, 
                                              self.parameters['max_clip'])] = \
                np.quantile(cyto_images, self.parameters['max_clip'])
        
        # Combine the images into a stack
        zero_images = np.zeros(dapi_images.shape)
        stacked_images = np.stack((dapi_images, cyto_images, dapi_images), axis=3)
        
        # Load the cellpose model. 'cyto2' performs better than 'cyto'.
        model = cellpose.models.Cellpose(gpu=self.parameters['use_gpu'], 
                                         model_type= self.parameters['model_type'])
        
        # Run the cellpose prediction
        masks, flows, styles, diams = model.eval(
            stacked_images, 
            diameter = self.parameters['diameter'], 
            do_3D = False, 
            flow_threshold = self.parameters['flow_threshold'],
            channels = [self.parameters['channel1'],self.parameters['channel2']], 
            resample = self.parameters['resample'], 
            normalize = self.parameters['normalize'])
        
        # if there is only one z plane
        if len(masks.shape) == 2:
            masks = masks.reshape(1, masks.shape[0], masks.shape[1])
          
        masks = self._connect_2D_masks_to_3D_masks(masks,
                                                   fragmentIndex,
                                                   n_neighbors = self.parameters['n_neighbors'],
                                                   distance_cutoff = self.parameters['connect_distance'])
                                                   
        if self.parameters['write_mask_images']:
            dapi_images = (dapi_images / dapi_images.max() * 255).astype(np.uint8)
            cyto_images = (cyto_images / cyto_images.max() * 255).astype(np.uint8)

            maskImages = np.array(flatten_list([[x,y,z] \
                for x,y,z in zip(masks, dapi_images, cyto_images)])
                ).astype(np.uint8)

            maskImageDescription = self.dataSet.analysis_tiff_description(
                    1, len(maskImages))

            with self.dataSet.writer_for_analysis_images(
                    self, 'mask', fragmentIndex) as outputTif:

                for maskImage in maskImages:
                    outputTif.save(
                            maskImage, 
                            photometric='MINISBLACK',
                            metadata=maskImageDescription)

        # identify features for each zplane sperately
        zposList = self.dataSet.get_data_organization().get_z_positions()

        # obtain the z index for each feature
        featureList = [ spatialfeature.SpatialFeature.feature_from_label_matrix(
            masks == label, fragmentIndex, 
            globalTask.fov_to_global_transform(fragmentIndex),
            list(np.unique(np.where(masks == label)[0]))) \
            for label in np.unique(masks) if label != 0 ]

        featureDB = self.get_feature_database()
        featureDB.write_features(featureList, fragmentIndex)

class CleanCellBoundaries(analysistask.ParallelAnalysisTask):
    
    '''
    A task to construct a network graph where each cell is a node, and overlaps
    are represented by edges. This graph is then refined to assign cells to the
    fov they are closest to (in terms of centroid). This graph is then refined
    to eliminate overlapping cells to leave a single cell occupying a given
    position.
    '''
    
    def __init__(self, dataSet, parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)

        self.segmentTask = self.dataSet.load_analysis_task(
            self.parameters['segment_task'])
        self.alignTask = self.dataSet.load_analysis_task(
            self.parameters['global_align_task'])

    def fragment_count(self):
        return len(self.dataSet.get_fovs())

    def get_estimated_memory(self):
        return 2048

    def get_estimated_time(self):
        return 30

    def get_dependencies(self):
        return [self.parameters['segment_task'],
                self.parameters['global_align_task']]

    def return_exported_data(self, fragmentIndex) -> nx.Graph:
        return self.dataSet.load_graph_from_gpickle(
            'cleaned_cells', self, fragmentIndex)

    def _run_analysis(self, fragmentIndex) -> None:
        allFOVs = np.array(self.dataSet.get_fovs())
        fovBoxes = self.alignTask.get_fov_boxes()
        fovIntersections = sorted([i for i, x in enumerate(fovBoxes) if
                                   fovBoxes[fragmentIndex].intersects(x)])
        intersectingFOVs = list(allFOVs[np.array(fovIntersections)])

        spatialTree = rtree.index.Index()
        count = 0
        idToNum = dict()
        for currentFOV in intersectingFOVs:
            cells = self.segmentTask.get_feature_database()\
                .read_features(currentFOV)
            cells = spatialfeature.simple_clean_cells(cells)

            spatialTree, count, idToNum = spatialfeature.construct_tree(
                cells, spatialTree, count, idToNum)

        graph = nx.Graph()
        cells = self.segmentTask.get_feature_database()\
            .read_features(fragmentIndex)
        cells = spatialfeature.simple_clean_cells(cells)
        graph = spatialfeature.construct_graph(graph, cells,
                                               spatialTree, fragmentIndex,
                                               allFOVs, fovBoxes)

        self.dataSet.save_graph_as_gpickle(
            graph, 'cleaned_cells', self, fragmentIndex)


class CombineCleanedBoundaries(analysistask.AnalysisTask):
    """
    A task to construct a network graph where each cell is a node, and overlaps
    are represented by edges. This graph is then refined to assign cells to the
    fov they are closest to (in terms of centroid). This graph is then refined
    to eliminate overlapping cells to leave a single cell occupying a given
    position.

    """
    def __init__(self, dataSet, parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)

        self.cleaningTask = self.dataSet.load_analysis_task(
            self.parameters['cleaning_task'])

    def get_estimated_memory(self):
        # TODO - refine estimate
        return 2048

    def get_estimated_time(self):
        # TODO - refine estimate
        return 5

    def get_dependencies(self):
        return [self.parameters['cleaning_task']]

    def return_exported_data(self):
        kwargs = {'index_col': 0}
        return self.dataSet.load_dataframe_from_csv(
            'all_cleaned_cells', analysisTask=self.analysisName, **kwargs)

    def _run_analysis(self):
        allFOVs = self.dataSet.get_fovs()
        graph = nx.Graph()
        for currentFOV in allFOVs:
            subGraph = self.cleaningTask.return_exported_data(currentFOV)
            graph = nx.compose(graph, subGraph)

        cleanedCells = spatialfeature.remove_overlapping_cells(graph)

        self.dataSet.save_dataframe_to_csv(cleanedCells, 'all_cleaned_cells',
                                           analysisTask=self)


class RefineCellDatabases(FeatureSavingAnalysisTask):
    def __init__(self, dataSet, parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)

        self.segmentTask = self.dataSet.load_analysis_task(
            self.parameters['segment_task'])
        self.cleaningTask = self.dataSet.load_analysis_task(
            self.parameters['combine_cleaning_task'])

    def fragment_count(self):
        return len(self.dataSet.get_fovs())

    def get_estimated_memory(self):
        # TODO - refine estimate
        return 2048

    def get_estimated_time(self):
        # TODO - refine estimate
        return 5

    def get_dependencies(self):
        return [self.parameters['segment_task'],
                self.parameters['combine_cleaning_task']]

    def _run_analysis(self, fragmentIndex):

        cleanedCells = self.cleaningTask.return_exported_data()
        originalCells = self.segmentTask.get_feature_database()\
            .read_features(fragmentIndex)
        featureDB = self.get_feature_database()
        cleanedC = cleanedCells[cleanedCells['originalFOV'] == fragmentIndex]
        cleanedGroups = cleanedC.groupby('assignedFOV')
        for k, g in cleanedGroups:
            cellsToConsider = g['cell_id'].values.tolist()
            featureList = [x for x in originalCells if
                           str(x.get_feature_id()) in cellsToConsider]
            featureDB.write_features(featureList, fragmentIndex)


class ExportCellMetadata(analysistask.AnalysisTask):
    """
    An analysis task exports cell metadata.
    """

    def __init__(self, dataSet, parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)
        
        self.segmentTask = self.dataSet.load_analysis_task(
            self.parameters['segment_task'])

    def get_estimated_memory(self):
        return 2048

    def get_estimated_time(self):
        return 30

    def get_dependencies(self):
        return [self.parameters['segment_task']]

    def _run_analysis(self):
        df = self.segmentTask.get_feature_database().read_feature_metadata()

        self.dataSet.save_dataframe_to_csv(df, 'feature_metadata',
                                           self.analysisName, index=False)
        
class ExportCellBoundaries(analysistask.AnalysisTask):
    """
    An analysis task exports cell boundaries as shapely file.
    """

    def __init__(self, dataSet, parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)
        
        self.segmentTask = self.dataSet.load_analysis_task(
            self.parameters['segment_task'])

    def get_estimated_memory(self):
        return 2048

    def get_estimated_time(self):
        return 30

    def get_dependencies(self):
        return [self.parameters['segment_task']]

    def _run_analysis(self):
        gdf = self.segmentTask.get_feature_database().read_feature_geopandas()
        self.dataSet.save_geodataframe_to_shp(gdf, 'feature_boundaries',
                                          self.analysisName, index=False)
        gdf_max_list = []
        for ft_id in set(gdf.id):
            indexMax = gdf[gdf.id == ft_id].area.argmax()
            gdf_max_list.append(gdf[gdf.id == ft_id].iloc[[indexMax]])
        gdf_max = pandas.concat(gdf_max_list, 0)

        self.dataSet.save_geodataframe_to_shp(gdf_max, 'feature_boundaries_max',
                                           self.analysisName, index=False)

