
# Install
```bash
mkdir -p $HOME/Github && cd $HOME/Github
git clone -b 3D-MERFISH https://github.com/rx3fang/MERlin.git
conda env create -f MERlin/merlin_env.yml
conda activate merlin_env
pip install -e ./MERlin
merlin -h

MERlin - the MERFISH decoding pipeline
usage: merlin [-h] [--profile] [--generate-only] [--configure] [-w ANALYSIS_DIR_NAME] [-a ANALYSIS_PARAMETERS] [-o DATA_ORGANIZATION] [-c CODEBOOK [CODEBOOK ...]]
              [-m MICROSCOPE_PARAMETERS] [-r MICROSCOPE_CHROMATIC_CORRECTIONS] [-l MICROSCOPE_ILLUMINATION_CORRECTIONS] [-d DEEPMERFISH_MODEL_NAME] [-p POSITIONS] [-n CORE_COUNT]
              [--check-done] [-t ANALYSIS_TASK] [-i FRAGMENT_INDEX] [-e DATA_HOME] [-s ANALYSIS_HOME] [-j PARAMETERS_HOME] [-k SNAKEMAKE_PARAMETERS] [--no_report NO_REPORT]
              dataset

Decode MERFISH data.

positional arguments:
  dataset               directory where the raw data is stored

optional arguments:
  -h, --help            show this help message and exit
  --profile             enable profiling
  --generate-only       only generate the directory structure and do not run any analysis.
  --configure           configure MERlin environment by specifying data, analysis, and parameters directories.
  -w ANALYSIS_DIR_NAME, --analysis-dir-name ANALYSIS_DIR_NAME
                        name of the output analysis folder to use
  -a ANALYSIS_PARAMETERS, --analysis-parameters ANALYSIS_PARAMETERS
                        name of the analysis parameters file to use
  -o DATA_ORGANIZATION, --data-organization DATA_ORGANIZATION
                        name of the data organization file to use
  -c CODEBOOK [CODEBOOK ...], --codebook CODEBOOK [CODEBOOK ...]
                        name of the codebook to use
  -m MICROSCOPE_PARAMETERS, --microscope-parameters MICROSCOPE_PARAMETERS
                        name of the microscope parameters to use
  -r MICROSCOPE_CHROMATIC_CORRECTIONS, --microscope-chromatic-corrections MICROSCOPE_CHROMATIC_CORRECTIONS
                        name of the microscope chromatic corrections to use
  -l MICROSCOPE_ILLUMINATION_CORRECTIONS, --microscope-illumination-corrections MICROSCOPE_ILLUMINATION_CORRECTIONS
                        name of the microscope illumination corrections to use
  -d DEEPMERFISH_MODEL_NAME, --deepmerfish-model-name DEEPMERFISH_MODEL_NAME
                        name of the deepmerfish model to use for image enhancement
  -p POSITIONS, --positions POSITIONS
                        name of the position file to use
  -n CORE_COUNT, --core-count CORE_COUNT
                        number of cores to use for the analysis
  --check-done          flag to only check if the analysis task is done
  -t ANALYSIS_TASK, --analysis-task ANALYSIS_TASK
                        the name of the analysis task to execute. If no analysis task is provided, all tasks are executed.
  -i FRAGMENT_INDEX, --fragment-index FRAGMENT_INDEX
                        the index of the fragment of the analysis task to execute
  -e DATA_HOME, --data-home DATA_HOME
                        the data home directory
  -s ANALYSIS_HOME, --analysis-home ANALYSIS_HOME
                        the analysis home directory
  -j PARAMETERS_HOME, --parameters-home PARAMETERS_HOME
                        the parameters directory
  -k SNAKEMAKE_PARAMETERS, --snakemake-parameters SNAKEMAKE_PARAMETERS
                        the name of the snakemake parameters file
  --no_report NO_REPORT
                        flag indicating that the snakemake stats should not be shared to improve MERlin
```
# Run MERlin on a local Linux server
```bash
# define data set variables
DATA_DIR_NAME=20230311-MFX.Disk.40X.WL-MB.100um-MOP/data
ANALYSIS_DIR_NAME=20230311-MFX.Disk.40X.WL-MB.100um-MOP/merlin_analysis
PARAMETERS_HOME=20230311-MFX.Disk.40X.WL-MB.100um-MOP/merlin_paramters
CORE_COUNT=5

# in the merlin_parameters/analysis/merlin_3D_decode.json file, change model_path to the pre-trained cellpose model in CellPoseSegment3D 
# "model_path": "20230311-MFX.Disk.40X.WL-MB.100um-MOP/merlin_paramters/cellpose_models/CP_20221125__disk_xy05um_z1um_DAPI_polyT"

merlin \
--analysis-dir-name $ANALYSIS_DIR_NAME \
--parameters-home $PARAMETERS_HOME \
--analysis-home $ANALYSIS_DIR_NAME \
--analysis-parameters merlin_3D_decode.json \
--microscope-parameters MERFISHX_disk_40X.json \
--data-organization dataorganization3D.csv \
--codebook M1_codebook_250.csv \
--positions tiled_positions_corrected.txt \
--core-count $CORE_COUNT \
--analysis-dir-name $ANALYSIS_DIR_NAME \
$DATA_DIR_NAME
```
