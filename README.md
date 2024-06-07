# Install
git clone -b 3D-MERFISH https://github.com/rx3fang/MERlin.git  
cd MERlin   
conda env create -f MERlin/merlin_env.yml    
conda activate merlin_env
pip install -e ./MERlin
merlin -h
