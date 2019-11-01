#!/bin/bash
#SBATCH --time=60:00:00
#SBATCH --account=rrg-sushama-ab
#SBATCH --mem-per-cpu=36864M      # memory; default unit is megabytes
#SBATCH --cpus-per-task=1

# Load Conda Environment with RPN libraries
. /home/cruman/run_conda.src

# Run Script
python wind-dist-high-res.py cPanCan011_675x540_SPN
#echo 2070-2099
#python wed-monthly.py 2070 2099

#echo 2040-2069
#python wed-monthly.py 2040 2069

#echo 2020-2049
#python wed-monthly.py 2020 2049

#echo 1976-2005
#python wed-monthly.py 1976 2005
