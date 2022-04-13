import os
import tifffile
import numpy as np
from merfishdecoder.core import zplane
from merfishdecoder.util import utilities
from merfishdecoder.util import segmentation

def run_job(
    dataSetName: str = None,
    fov: int = None,
    segmentedImagesDir: str = None,
    outputName: str = None):
    
    """
    Extract features.

    Args
    ----
    dataSetName: input dataset name.

    fov: the field of view to be processed.
    
    segmentedImagesDir: directory that contains the 
        segmented imagess. 
    
    outputName: output file name for segmented images.
    
    """
    
    # dataSetName = "MERFISH_test/data"
    # fov = 0
    # outputName = "segmentedImages/fov_0_zpos_3.0.npz"
    # warpedImagesName = "warpedImages/fov_0_zpos_3.0.tif"
    # featureName = "DAPI"
    
    # check points
    utilities.print_checkpoint("Image Segmentation")
    utilities.print_checkpoint("Start")

    # create a zplane object
    zp = zplane.Zplane(dataSetName,
                       fov = fov,
                       zpos = zpos)

    # create the folder
    os.makedirs(os.path.dirname(outputName),
        exist_ok=True)
    
    # load readout images
    zp.load_warped_images(
        warpedImagesName)
    
    # run cell pose for cell segmentation
    mask = segmentation.run_cell_pose(
        gpu = gpu,
        modelType = modelType,
        images = zp.get_readout_images([featureName])[0],
        diameter = diameter
        ).astype(np.uint16)    
    
    np.savez_compressed(file=outputName,
                        mask=mask)
    
    utilities.print_checkpoint("Done")

