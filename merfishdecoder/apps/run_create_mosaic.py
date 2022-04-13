#!/usr/bin/python
# ----------------------------------------------------------------------------------------
# An application to create mosiac image.
# ----------------------------------------------------------------------------------------
# Rongxin Fang
# latest update: 04/17/21
# r4fang@gmail.com
# ----------------------------------------------------------------------------------------

import os
import sys
import cv2
import tifffile
import skimage
import random
import collections
import numpy as np
from skimage import registration

from merfishdecoder.core import zplane
from merfishdecoder.core import dataset
from merfishdecoder.util import registration
from merfishdecoder.util import imagefilter
from merfishdecoder.util import utilities
from merfishdecoder.util import segmentation


class SimpleGlobalAlignment():

    """A global alignment that uses the theoretical stage positions in
    order to determine the relative positions of each field of view.
    """

    def __init__(self, dataSet, parameters=None, analysisName=None):
        self.dataSet = dataSet

    def get_estimated_memory(self):
        return 1

    def get_estimated_time(self):
        return 0

    def _run_analysis(self):
        # This analysis task does not need computation
        pass

    def get_dependencies(self):
        return []

    def fov_coordinates_to_global(self, fov, fovCoordinates):
        fovStart = self.dataSet.get_fov_offset(fov)
        micronsPerPixel = self.dataSet.get_microns_per_pixel()
        if len(fovCoordinates) == 2:
            return (fovStart[0] + fovCoordinates[0]*micronsPerPixel,
                    fovStart[1] + fovCoordinates[1]*micronsPerPixel)
        elif len(fovCoordinates) == 3:
            zPositions = self.dataSet.get_z_positions()
            return (np.interp(fovCoordinates[0], np.arange(len(zPositions)),
                              zPositions),
                    fovStart[0] + fovCoordinates[1]*micronsPerPixel,
                    fovStart[1] + fovCoordinates[2]*micronsPerPixel)

    def fov_coordinate_array_to_global(self, fov: int,
                                       fovCoordArray: np.array) -> np.array:
        tForm = self.fov_to_global_transform(fov)
        toGlobal = np.ones(fovCoordArray.shape)
        toGlobal[:, [0, 1]] = fovCoordArray[:, [1, 2]]
        globalCentroids = np.matmul(tForm, toGlobal.T).T[:, [2, 0, 1]]
        globalCentroids[:, 0] = fovCoordArray[:, 0]
        return globalCentroids

    def fov_global_extent(self, fov: int):
        """
        Returns the global extent of a fov, output interleaved as
        xmin, ymin, xmax, ymax

        Args:
            fov: the fov of interest
        Returns:
            a list of four floats, representing the xmin, xmax, ymin, ymax
        """

        return [x for y in (self.fov_coordinates_to_global(fov, (0, 0)),
                            self.fov_coordinates_to_global(fov, (2048, 2048)))
                for x in y]

    def global_coordinates_to_fov(self, fov, globalCoordinates):
        tform = np.linalg.inv(self.fov_to_global_transform(fov))

        def convert_coordinate(coordinateIn):
            coords = np.array([coordinateIn[0], coordinateIn[1], 1])
            return np.matmul(tform, coords).astype(int)[:2]
        pixels = [convert_coordinate(x) for x in globalCoordinates]
        return pixels

    def fov_to_global_transform(self, fov):
        micronsPerPixel = self.dataSet.get_microns_per_pixel()
        globalStart = self.fov_coordinates_to_global(fov, (0, 0))

        return np.float32([[micronsPerPixel, 0, globalStart[0]],
                           [0, micronsPerPixel, globalStart[1]],
                           [0, 0, 1]])

    def get_global_extent(self):
        fovSize = self.dataSet.get_image_dimensions()
        fovBounds = [self.fov_coordinates_to_global(x, (0, 0))
                     for x in self.dataSet.get_fovs()] + \
                    [self.fov_coordinates_to_global(x, fovSize)
                     for x in self.dataSet.get_fovs()]

        minX = np.min([x[0] for x in fovBounds])
        maxX = np.max([x[0] for x in fovBounds])
        minY = np.min([x[1] for x in fovBounds])
        maxY = np.max([x[1] for x in fovBounds])

        return minX, minY, maxX, maxY

class CorrelationGlobalAlignment():

    """
    A global alignment that uses the cross-correlation between
    overlapping regions in order to determine the relative positions
    of each field of view.
    """

    # TODO - implement.  I expect rotation might be needed for this alignment
    # if the x-y orientation of the camera is not perfectly oriented with
    # the microscope stage

    def __init__(self, dataSet, parameters=None, analysisName=None):
        self.dataSet = dataSet

    def get_estimated_memory(self):
        return 1000

    def get_estimated_time(self):
        return 60

    def fov_coordinates_to_global(self, fov, fovCoordinates):
        raise NotImplementedError

    def fov_to_global_transform(self, fov):
        raise NotImplementedError

    def get_global_extent(self):
        raise NotImplementedError

    def fov_coordinate_array_to_global(self, fov: int,
                                       fovCoordArray: np.array) -> np.array:
        raise NotImplementedError

    @staticmethod
    def _calculate_overlap_area(x1, y1, x2, y2, width, height):
        """Calculates the overlapping area between two rectangles with
        equal dimensions.
        """

        dx = min(x1+width, x2+width) - max(x1, x2)
        dy = min(y1+height, y2+height) - max(y1, y2)

        if dx > 0 and dy > 0:
            return dx*dy
        else:
            return 0

    def _get_overlapping_regions(self, fov: int, minArea: int = 2000):
        """Get a list of all the fovs that overlap with the specified fov.
        """
        positions = self.dataSet.get_stage_positions()
        pixelToMicron = self.dataSet.get_microns_per_pixel()
        fovMicrons = [x*pixelToMicron
                      for x in self.dataSet.get_image_dimensions()]
        fovPosition = positions.loc[fov]
        overlapAreas = [i for i, p in positions.iterrows()
                        if self._calculate_overlap_area(
                p['X'], p['Y'], fovPosition['X'], fovPosition['Y'],
                fovMicrons[0], fovMicrons[1]) > minArea and i != fov]

        return overlapAreas

    def _run_analysis(self):
        fov1 = self.dataSet.get_fiducial_image(0, 0)
        fov2 = self.dataSet.get_fiducial_image(0, 1)

        return fov1, fov2

def _micron_to_mosaic_transform(
    micronExtents: tuple,
    mosaicMicronsPerPixel: float
    ) -> np.ndarray:
    s = 1 / mosaicMicronsPerPixel
    return np.float32(
            [[s*1, 0, -s*micronExtents[0]],
             [0, s*1, -s*micronExtents[1]],
             [0, 0, 1]])

def _micron_to_mosaic_pixel(micronCoordinates,
                            micronExtents,
                            mosaicMicronsPerPixel
                           ) -> np.ndarray:
    """
    Calculates the mosaic coordinates in pixels from the specified
    global coordinates.
    """

    return np.matmul(_micron_to_mosaic_transform(micronExtents, mosaicMicronsPerPixel),
                     np.append(micronCoordinates, 1)).astype(np.int32)[:2]

def _transform_image_to_mosaic(
        inputImage: np.ndarray,
        fov: int,
        alignTask,
        micronExtents,
        mosaicDimensions: tuple,
        mosaicMicronsPerPixel: float
        ) -> np.ndarray:
    transform = \
            np.matmul(_micron_to_mosaic_transform(micronExtents, mosaicMicronsPerPixel),
                      alignTask.fov_to_global_transform(fov))
    return cv2.warpAffine(
            inputImage, transform[:2, :], mosaicDimensions)

class merfishTask:

    """
    A object for merfish task.

    """

    def __init__(self, **arguments):
        for (arg, val) in arguments.items():
            setattr(self, arg, val)

    def to_string(self):
        return ("\n".join(["%s = %s" % (str(key), str(val)) \
                for (key, val) in self.__dict__.items() ]))

    def run_job(self):
        dataSetName = self.dataSetName
        featureName = self.featureName
        cropWidth = self.cropWidth
        mosaicMicronsPerPixel = self.mosaicMicronsPerPixel
        zpos = self.zpos
        highPassFilterSigma = self.highPassFilterSigma
        refFrameIndex = self.refFrameIndex
        outputName = self.outputName
        
        utilities.print_checkpoint(self.to_string() + "\n")
        utilities.print_checkpoint("Create Mosaic Image")
        utilities.print_checkpoint("Start")

        dataSet = dataset.MERFISHDataSet(
            dataDirectoryName = dataSetName)
        os.chdir(dataSet.analysisPath)
        
        alignTask = SimpleGlobalAlignment(dataSet=dataSet)
        micronExtents = alignTask.get_global_extent()

        mosaicDimensions = tuple(_micron_to_mosaic_pixel(
                micronExtents[-2:], micronExtents, mosaicMicronsPerPixel))
                
        mosaic = np.zeros(np.flip(mosaicDimensions, axis=0), dtype=np.uint16)
        for fov in dataSet.get_fovs():
            print("fov: %d" % fov)
            featureImages = []
            zp = zplane.Zplane(dataSetName,
                fov=fov, zpos=zpos)
            frameNames = [ zp.get_readout_name()[refFrameIndex], featureName]
            zp.load_readout_images(frameNames)
            #(zp, errors) = registration.correct_drift(
            #    obj = zp,
            #    frameNames = frameNames,
            #    refFrameIndex = refFrameIndex,
            #    highPassSigma = highPassFilterSigma)
            featureImages.append(
                zp.get_readout_image_from_readout_name(featureName))
            inputImage = np.array(featureImages).max(axis=0)

            if cropWidth > 0:
                inputImage[:cropWidth, :] = 0
                inputImage[inputImage.shape[0] - cropWidth:, :] = 0
                inputImage[:, :cropWidth] = 0
                inputImage[:, inputImage.shape[0] - cropWidth:] = 0
            
            if self.drawFovLabels:
                inputImage = cv2.putText(inputImage, str(fov),
                        (int(0.2*inputImage.shape[0]),
                        int(0.2*inputImage.shape[1])),
                        0, 10, (65000, 65000, 65000), 20)

            transformedImage = _transform_image_to_mosaic(
                inputImage, fov, alignTask, micronExtents,
                mosaicDimensions, mosaicMicronsPerPixel)

            divisionMask = np.bitwise_and(
                transformedImage > 0, mosaic > 0)

            mosaic = cv2.add(mosaic, transformedImage, dst=mosaic,
                    mask=np.array(
                        transformedImage > 0).astype(np.uint8))

            dividedMosaic = cv2.divide(mosaic, 2)
            mosaic[divisionMask] = dividedMosaic[divisionMask]

        tifffile.imwrite(
            data = mosaic.astype(np.uint16),
            file= outputName)
        
        utilities.print_checkpoint("Done")
    
def main():

    mt = merfishTask(
        dataSetName = sys.argv[1],
        featureName = sys.argv[2],
        cropWidth = int(sys.argv[3]),
        mosaicMicronsPerPixel = float(sys.argv[4]),
        zpos = float(sys.argv[5]),
        outputName = sys.argv[6],
        drawFovLabels = bool(sys.argv[7]),
        refFrameIndex = 0,
        highPassFilterSigma = 3)

    mt.run_job()

if __name__ == "__main__":
    main()

