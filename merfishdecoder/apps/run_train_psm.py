import glob, os
import pickle
import pandas as pd
import numpy as np
import random

from sklearn import svm
from merfishdecoder.core import dataset
from merfishdecoder.util import utilities
from merfishdecoder.util import barcoder

def run_job(dataSetName: str = None,
            decodedImagesDir: str = None,
            outputName: str = "pixel_score_machine.pkl",
            zposNum: int = 50):
    
    """
    Training pixel-based score machine.

    Args
    ----
    dataSetName: input dataset name.
    
    decodedImagesName: a list of decoded image file names.

    outputName: pixel scoring model file name.
    
    zposNum: number of zplanes used for training the model.
             
    """
    
    # dataSetName = "MERFISH_test/data/"
    # decodedImagesDir = "decodedImages"
    # outputName = "pixel_score_machine.pkl"
    
    utilities.print_checkpoint("Train Pixel Score Model")
    utilities.print_checkpoint("Start")
    
    # generate dataset object
    dataSet = dataset.MERFISHDataSet(
        dataDirectoryName = dataSetName);

    # change to work directory
    os.chdir(dataSet.analysisPath)
    fnames = glob.glob(os.path.join(decodedImagesDir, "*.npz"))
    fnames = random.sample(fnames,
        min(len(fnames), zposNum))
    
    X_tr = []; Y_tr = [];
    for fn in fnames:
        decodes = np.load(fn)

        m = np.log10(decodes["magnitudeImage"][
            decodes["decodedImage"] > -1])

        d = decodes["distanceImage"][
            decodes["decodedImage"] > -1]
        
        o = decodes["decodedImage"][
            decodes["decodedImage"] > -1]
        
        y = np.ones(m.shape[0])
        y[np.isin(o, dataSet.get_codebook().get_blank_indexes())] = 0
        
        x = np.array([m, d]).T

        # now sample the same number of positive barcodes
        numPos = np.count_nonzero(y == 1)
        numNeg = np.count_nonzero(y == 0)

        # sample
        num = min(numPos, numNeg)
        idxPos = np.random.choice(np.nonzero(y)[0], num)
        idxNeg = np.random.choice(np.where(y == 0)[0], num)

        # create training dataset
        x_tr = x[np.concatenate([idxPos, idxNeg])]
        y_tr = y[np.concatenate([idxPos, idxNeg])]
        
        X_tr.append(x_tr)
        Y_tr.append(y_tr)
        decodes.close()

    X_tr = np.concatenate(X_tr)
    Y_tr = np.concatenate(Y_tr)
    
    idx = np.random.choice(
        range(X_tr.shape[0]), 
        min(X_tr.shape[0], 50000))

    rbf_svc = svm.SVC(
        kernel='rbf', 
        probability=True, 
        gamma="auto",
        C = 0.5)

    rbf_svc.fit(X_tr[idx], Y_tr[idx])
    
    pickle.dump(rbf_svc, 
        open(outputName, 'wb'))
    
    dat = pd.DataFrame(X_tr[idx], columns=["m", "d"])
    dat = dat.assign(y = Y_tr[idx])
    dat = dat.assign(p = rbf_svc.predict_proba(X_tr[idx])[:,1])
    dat.to_csv("pixel_score_machine.csv", index=False)
    utilities.print_checkpoint("Done")


