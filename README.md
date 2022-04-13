## merfishdecoder
A MERFISH decoding pipeline.

## Requirements
* Linux/Unix
* Python (3.6)

## Installation

```bash
$ wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
$ bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda
$ echo ". $HOME/miniconda/etc/profile.d/conda.sh" >> ~/.bashrc
$ echo "conda activate" >> ~/.bashrc
$ source .bashrc
$ conda activate base
$ conda config --set always_yes true
$ conda config --set quiet true
$ conda create -n md_env python=3.6
$ source activate md_env
$ conda install rtree pytables
$ conda install -c conda-forge sharedarray
$ pip install -e merfishdecoder
$ printf 'DATA_HOME=/RawData/MERFISH_raw_data/\nANALYSIS_HOME=/Analysis/MERFISH/merfish_analysis/\nPARAMETERS_HOME=~/merfishdecoder/merfish-parameters/' >~/.md_env
```

## Example

```bash
$ snakemake -j 10 \
	--snakefile merfish_parameters/snakemake/SnakefilePSM \
	--configfile merfish_parameters/analysis/20200303_hMTG_V11_4000gene_best_sample.yaml \
	--cluster-config merfish_parameters/clusters/cluster.json \
	--cluster "sbatch -p {cluster.partition} -N {cluster.node} --mem={cluster.mem} -t {cluster.time} -o {cluster.out} -e {cluster.err}" \
	--latency-wait 120 \
	--restart-times 2

$ snakemake -j 100 \
	--snakefile ~/merfish_parameters/snakemake/Snakefile \
	--configfile ~/merfish_parameters/analysis/20200303_hMTG_V11_4000gene_best_sample.yaml \
	--cluster-config ~/merfish_parameters/clusters/cluster.json \
	--cluster "sbatch -p {cluster.partition} -N {cluster.node} --mem={cluster.mem} -t {cluster.time} -o {cluster.out} -e {cluster.err}" \
	--latency-wait 120 \
	--restart-times 2
```



