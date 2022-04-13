import os
import glob
import pandas as pd
import numpy as np

from merfishdecoder.core import dataset

def FileCheck(fn):
	try:
		pd.read_hdf(fn)
		return True
	except:
		return False

def estimate_one_to_zero_err(fn, cb):
    cbMat = cb.get_barcodes()
    
    cbMatMagnitudes = np.array([ np.linalg.norm(cbMat[i]) \
       for i in range(cbMat.shape[0]) ], dtype=np.float32)
    cbMatMagnitudes[cbMatMagnitudes == 0] = 1
    cbMatNorm = cbMat / cbMatMagnitudes[:, None]
    x = pd.read_hdf(fn)
    m = np.array(x[cb.get_bit_names()])
    m = m / np.array(x.magnitudes)[:, None]
    return 1 - (m * cbMat[x.barcode_id]).sum(axis=0) / \
        cbMatNorm[x.barcode_id].sum(axis=0)

def estimate_zero_to_one_err(fn, cb):
    cbMat = cb.get_barcodes()
    
    cbMatMagnitudes = np.array([ np.linalg.norm(cbMat[i]) \
       for i in range(cbMat.shape[0]) ], dtype=np.float32)
    cbMatMagnitudes[cbMatMagnitudes == 0] = 1
    cbMatNorm = cbMat / cbMatMagnitudes[:, None]
    
    x = pd.read_hdf(fn)
    m = np.array(x[cb.get_bit_names()])
    m = m / np.array(x.magnitudes)[:, None]
    return (m * (1 - cbMat[x.barcode_id])).sum(axis=0) / \
        (cbMatNorm.max() - cbMatNorm[x.barcode_id]).sum(axis=0)
  
def run_job(dataSetName: str = None,
            pixelTracesDir: str = None,
            outputName: str = None):
    
    # dataSetName = "MERFISH_test/data"
    # pixelTracesDir = "extractedPixelTraces"
    # outputName = "qualityControls/bitErrors.csv"
    
    dataSet = dataset.MERFISHDataSet(dataSetName)
    os.chdir(dataSet.analysisPath)
    cb = dataSet.get_codebook()
    
    # create output folder
    os.makedirs(os.path.dirname(outputName),
                exist_ok=True)

    fnames = [ x for x in glob.glob(pixelTracesDir + "/*h5") \
        if FileCheck(x) ]
    
    oneToZeroErr = np.array([ estimate_one_to_zero_err(fn, cb) \
        for fn in fnames ]).mean(axis=0)
    zeroToOneErr = np.array([ estimate_zero_to_one_err(fn, cb) \
        for fn in fnames ]).mean(axis=0)

    dat = pd.DataFrame({
    	"oneToZeroErr": oneToZeroErr, 
    	"zeroToOneErr": zeroToOneErr
    	}, index = cb.get_bit_names())
    
    dat.to_csv(outputName)
