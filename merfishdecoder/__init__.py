# this code was copied and modified from original MERlin package
import dotenv
import os
import glob
import json
import importlib
from typing import List

envPath = os.path.join(os.path.expanduser('~'), '.merfishdecoderenv')

if os.path.exists(envPath):
    dotenv.load_dotenv(envPath)

    try:
        DATA_HOME = os.path.expanduser(os.environ.get('DATA_HOME'))
        ANALYSIS_HOME = os.path.expanduser(os.environ.get('ANALYSIS_HOME'))
        PARAMETERS_HOME = os.path.expanduser(os.environ.get('PARAMETERS_HOME'))
        ANALYSIS_PARAMETERS_HOME = os.sep.join(
                [PARAMETERS_HOME, 'analysis'])
        CODEBOOK_HOME = os.sep.join(
                [PARAMETERS_HOME, 'codebooks'])
        DATA_ORGANIZATION_HOME = os.sep.join(
                [PARAMETERS_HOME, 'dataorganization'])
        POSITION_HOME = os.sep.join(
                [PARAMETERS_HOME, 'positions'])
        MICROSCOPE_PARAMETERS_HOME = os.sep.join(
                [PARAMETERS_HOME, 'microscope'])
        FPKM_HOME = os.sep.join([PARAMETERS_HOME, 'fpkm'])
        SNAKEMAKE_PARAMETERS_HOME = os.sep.join(
            [PARAMETERS_HOME, 'snakemake'])

    except TypeError:
        print('merfishdecoder environment appears corrupt. Please run ' +
              '\'merfishdecoder --configure .\' in order to configure the environment.')
else:
    print(('Unable to find merfishdecoder environment file at %s. Please run ' +
          '\'merfishdecoder --configure .\' in order to configure the environment.')
          % envPath)


def store_env(dataHome, analysisHome, parametersHome):
    with open(envPath, 'w') as f:
        f.write('DATA_HOME=%s\n' % dataHome)
        f.write('ANALYSIS_HOME=%s\n' % analysisHome)
        f.write('PARAMETERS_HOME=%s\n' % parametersHome)


class IncompatibleVersionException(Exception):
    pass


def version():
    import pkg_resources
    return "v0.0"; #pkg_resources.require("merfishdecoder")[0].version


def is_compatible(testVersion: str, baseVersion: str = None) -> bool:
    """ Determine if testVersion is compatible with baseVersion

    Args:
        testVersion: the version identifier to test, as the string 'x.y.z'
            where x is the major version, y is the minor version,
            and z is the patch.
        baseVersion: the version to check testVersion's compatibility. If  not
            specified then the current MERlin version is used as baseVersion.
    Returns: True if testVersion are compatible, otherwise false.
    """
    if baseVersion is None:
        baseVersion = version()
    return testVersion.split('.')[0] == baseVersion.split('.')[0]
