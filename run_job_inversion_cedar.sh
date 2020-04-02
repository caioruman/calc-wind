#!/bin/bash
#SBATCH --time=14:00:00
#SBATCH --account=def-sushama-ab
#SBATCH --mem-per-cpu=6144M      # memory; default unit is megabytes
#SBATCH --cpus-per-task=1

# load base  python modules
#. /project/6001700/huziy/Software/software/2017/Core/miniconda3/4.3.27/bin/activate root
. /home/cruman/projects/rrg-sushama-ab/cruman/miniconda/bin/activate py3.7
#sleep 30

# add my dependencies
# Graham
#export PYTHONPATH=/home/huziy/project/huziy/Python/Projects/RPN/src:${PYTHONPATH}
# Cedar
export PYTHONPATH=/home/huziy/rrg-sushama-ab/Python/Projects/RPN/src:${PYTHONPATH}
#sleep 30
export LD_LIBRARY_PATH=~huziy/lib:${LD_LIBRARY_PATH}
#sleep 30


# add pylibrmn dependencies
. s.ssmuse.dot pylibrmn_deps
# export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:~huziy/lib


# launch the script

python /home/cruman/scratch/Scripts/calc-wind/wind-distrcp45.py 2070 2085

# in_nc_file=/home/poitras/ddm/directions_WestCaUs_dx0.11deg.nc3
# out_fst_file=/scratch/huziy/directions_WestCaUs_dx0.11deg.fst
# nml_file=

#in_nc_file=/home/poitras/ddm/directions_CanArc_0.04deg.nc
#out_fst_file=/scratch/huziy/directions_CanArc_0.04deg.fst
#nml_file=/home/poitras/ECCC/CanArc_004deg_1080x900/gem_settings.nml

