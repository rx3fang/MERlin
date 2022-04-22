set -e

merlin -a merlin_decode_w_opt.json \
        -m MERFISHX_40X.json \
        -o 20220419-MFX.DISK.40X.oil-MB.10um-250gene.AP_dataorganization_part1_1000ms.csv \
        -c M1_codebook_250.csv \
        -p 20220419-MFX.DISK.40X.oil-MB.10um-250gene.AP_tiled_positions1.txt \
        -k snakemake_norm_22bits_6z.json \
		MERFISHX/20220419-MFX.DISK.40X.oil-MB.10um-250gene.AP/data
