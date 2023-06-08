conda create --name merlin_3.8 python=3.8.13 -c conda-forge
conda activate merlin_3.8
conda install ipykernel -c conda-forge
conda install rtree=0.9.4 -c conda-forge
conda install pytables=3.6.1 -c conda-forge
conda install shapely=1.6.4 -c conda-forge

# pip install --upgrade pip setuptools wheel
pip install opencv-python==4.5.5.64
pip install cellpose==2.0.5
pip install -e MERlin

Check ~/.merlinenv
DATA_HOME=gc://r3fang_east4/merfish_raw_data
ANALYSIS_HOME=/home/r3fang_g_harvard_edu/merlin_analysis
PARAMETERS_HOME=/home/r3fang_g_harvard_edu/merlin_parameters

# guava
DATA_HOME=/home/r3fang/NAS/Fang/RawData/MERFISH2
ANALYSIS_HOME=/home/r3fang/NAS/Fang/r3fang/MERFISH/merlin_analysis
PARAMETERS_HOME=/home/r3fang/NAS/Fang/r3fang/MERFISH/merlin_parameters
