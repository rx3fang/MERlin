import argparse
import os, sys
import pickle
import pandas as pd
import numpy as np
import geopandas as geo

from merfishdecoder.core import zplane
from merfishdecoder.util import registration
from merfishdecoder.util import imagefilter
from merfishdecoder.util import utilities
from merfishdecoder.util import segmentation

def main():
    parser = argparse.ArgumentParser(description='MERFISH Analysis.')
    
    parser_req = parser.add_argument_group("required inputs")
    parser_req.add_argument("--data-set-name",
                             type=str,
                             required=True,
                             help="MERFISH dataset name.")

    parser_req.add_argument("--fov",
                             type=int,
                             required=True,
                             help="Field of view.")

    parser_req.add_argument("--zpos",
                             type=float,
                             required=True,
                             help="Z plane.")

    parser_req.add_argument("--output-name",
                             type=str,
                             required=True,
                             help="Output name.")
    
    parser_req.add_argument("--feature-name",
                             type=str,
                             required=True,
                             help="Feature name.")

    parser_req.add_argument("--feature-diameter",
                             type=int,
                             required=True,
                             help="Feature diameter.")

    parser_req.add_argument("--feature-model-type",
                             type=str,
                             required=True,
                             help="Feature model type.")
    
    parser_opt = parser.add_argument_group("optional inputs")
    parser_opt.add_argument("--ref-frame-index",
                            type=int,
                            default=0,
                            help="Reference frame index for correcting drift.")

    parser_opt.add_argument("--high-pass-filter-sigma",
                            type=int,
                            default=3,
                            help="Low pass sigma for high pass filter prior to registration.")

    args = parser.parse_args()

    dataSetName = args.data_set_name
    fov = args.fov
    zpos = args.zpos
    outputName = args.output_name
    
    featureName = args.feature_name
    modelType = args.feature_model_type
    featureDiameter = args.feature_diameter
    
    refFrameIndex = args.ref_frame_index
    highPassFilterSigma = args.high_pass_filter_sigma
    
    utilities.print_checkpoint("Start Segmentation")
    
    # generate zplane object
    zp = zplane.Zplane(dataSetName,
                       fov=fov,
                       zpos=zpos)
    
    # create the folder
    os.makedirs(os.path.dirname(outputName),
        exist_ok=True)
    
    utilities.print_checkpoint("Load Readout Images")
    
    # load readout images
    frameNames = [ zp.get_readout_name()[refFrameIndex], featureName]
    zp.load_readout_images(frameNames)
    
    utilities.print_checkpoint("Correct Stage Drift")
    (zp, errors) = registration.correct_drift(
        obj = zp,
        frameNames = frameNames,
        refFrameIndex = refFrameIndex,
        highPassSigma = highPassFilterSigma)
    
    utilities.print_checkpoint("Correct Chromatic Aberration")
    profile = zp.get_chromatic_aberration_profile()

    zp = registration.correct_chromatic_aberration(
        obj = zp,
        frameNames = frameNames,
        profile = profile)
    
    utilities.print_checkpoint("Cellpose Segmentation")
    
    # run cell pose for cell segmentation
    segmentedImage = segmentation.run_cell_pose(
        modelType = modelType,
        images = [ zp.get_readout_images([featureName]) ],
        diameter = featureDiameter
        )[0]

    # extract the feature
    utilities.print_checkpoint("Extract Features")
    ft = [ (idx, segmentation.extract_polygon_per_index(segmentedImage, idx)) \
        for idx in np.unique(segmentedImage[segmentedImage > 0]) ]
    ft = [ (i, x) for (i, x) in ft if x != None ]

    # convert to a data frame
    if len(ft) > 0:
        features = geo.GeoDataFrame(
            pd.DataFrame({
                "fov": [fov] * len(ft),
                "global_z": [zpos] * len(ft),
                "z": zp._dataSet.get_z_positions().index(zpos)}),
            geometry=[x[1] for x in ft])
    else:
        features = geo.GeoDataFrame(
            pd.DataFrame(columns = ["fov", "global_z", "z", "x", "y"]), 
            geometry=None)
    
    if not features.empty:
        features = features.assign(x = features.centroid.x)
        features = features.assign(y = features.centroid.y)
        features.to_file(outputName)
    else:
        with open(outputName, 'w') as fp: 
            pass

    utilities.print_checkpoint("Done")

if __name__ == "__main__":
    main()
