## MERlin (v0.3.4)
A MERFISH decoding pipeline.

## Requirements
* Linux/Unix
* Python (3.6)

## Installation

```bash
$ git clone https://github.com/r3fang/MERlin.git
$ conda create --name merlin python=3.6.10
$ conda activate merlin
$ conda install rtree=0.9.4
$ conda install pytables=3.6.1
$ conda install shapely=1.6.4
$ pip install opencv-python-headless
$ pip install cellpose
$ pip --no-cache-dir install -e MERlin
```

## Example

```bash
merlin -a analysis.json \
	-m microscope.json \
	-o dataorganization.csv \
	-c codebook.csv \
	-p titled_positions.txt \
	-k snakemake.json \
	-r chromatic_aberration_profile.pkl \
	-l illumination_aberration_profile.pkl \
	dataset
```
