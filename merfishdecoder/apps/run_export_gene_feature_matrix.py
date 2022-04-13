import os
import scipy
import sys

from scipy import io
from scipy import sparse
import pandas as pd
import geopandas as geo
import numpy as np
import multiprocessing as mp

from merfishdecoder.core import dataset
from merfishdecoder.util import utilities

def read_barcodes_per_fov(
    fname: str = None,
    fov: int = None
    ) -> pd.core.frame.DataFrame:

    """
    Read barcodes from a single field of view.
    """
    try:
        return pd.concat([
            pd.read_hdf(fname, key="fov_%d" % fov) ],
            axis=1)
    except KeyError:
        print("fov_%d does not exist" % fov)
        return None

def export_gene_feature_vector_per_fov(
    barcodesName: str = None,
    fov: int = None
    ) -> dict:
    
    bd = read_barcodes_per_fov(
        barcodesName, fov)
    if bd is None:
        return dict()
    bd = bd[bd.feature_name != "NA"] 
    fnList = np.unique(bd["feature_name"])
    return dict([ (fn, np.unique(
        bd[bd.feature_name == fn].barcode_id,
        return_counts=True)) \
        for fn in fnList ])

def run_job(dataSetName: str = None,
            barcodesName: str = None,
            featuresName: str = None,
            outputName: str = None,
            maxCores: int = 1):
    
    utilities.print_checkpoint("Export gene feature matrix")
    utilities.print_checkpoint("Start")
    
    
    dataSet = dataset.MERFISHDataSet(
            dataSetName)

    # change working directory
    os.chdir(dataSet.analysisPath)

    # create foder
    os.makedirs(outputName, 
                exist_ok=True)
    
    # read features
    features = geo.read_file(
        featuresName) 
    
    # assign global x and y
    features = features.assign(global_x = features.centroid.x)
    features = features.assign(global_y = features.centroid.y)

    # create the centroid for each feature
    features = pd.DataFrame([[ 
        features[features.name == fn].fov.mean(),
        features[features.name == fn].x.mean(),
        features[features.name == fn].y.mean(),
        features[features.name == fn].global_x.mean(),
        features[features.name == fn].global_y.mean(),
        fn,
        features[features.name == fn].area.sum(),
        features[features.name == fn].area.mean()]
        for fn in np.unique(features.name) ],
        columns = ["fov", "x", "y", "global_x", \
            "global_y", "name", "area", "avg_area"])

    # assign barcodes to cell
    with mp.Pool(processes=maxCores) as pool:
        vectorList = pool.starmap(
            export_gene_feature_vector_per_fov,
            (( barcodesName, fov) \
            for fov in dataSet.get_fovs()))
    vectorList = {k: v for d in vectorList \
        for k, v in d.items()}

    # create the cell_gene_matrix
    featureNum = features.shape[0]
    geneNum = dataSet.get_codebook().get_barcode_count()
    print("matrix dimention = %d x %d" % (featureNum, geneNum))

    mat = np.zeros((featureNum, geneNum))
    for key in vectorList:
        (idx, count) = vectorList[key]
        if key in np.array(features.name):
            mat[list(features.name).index(key), idx.astype(int)] += count

    scipy.io.mmwrite(
        os.path.join(outputName, "matrix.mtx"),
        sparse.csr_matrix(mat.T.astype(np.float32)))
    features.to_csv(
        os.path.join(outputName, "features.tsv"),
        index = False, header = True, sep='\t')
    dataSet.get_codebook().get_data()[["id", "name"]].to_csv(
        os.path.join(outputName, "genes.tsv"),
        index = False, header = False, sep='\t')
    
    utilities.print_checkpoint("Done")

def main():
    dataSetName = sys.argv[1]
    barcodesName = sys.argv[2]
    featuresName = sys.argv[3]
    outputName = sys.argv[4]
    maxCores = int(sys.argv[5])
    run_job(dataSetName, barcodesName, featuresName, outputName, maxCores)
    
if __name__ == "__main__":
    main()


