import numpy as np
import pandas as pd
import sys

from datetime import date, datetime, timedelta

from glob import glob
from rpn.rpn import RPN
from rpn.domains.rotated_lat_lon import RotatedLatLon
from rpn import level_kinds

from netCDF4 import Dataset
import time

'''
    Using the samples files of the simulation. (3 hourly)
    - Using the bottom level wind from the dp file (~34m wind)
    - Using the bottom level temperature from the dp file (~34m temperature) and the skin temperature (I0? J8?)
      - Also calculate the 925-1000 inversion and compare it
    - Sensible Heat Flux (AH)
'''

def main():

    print("hello world")

    #/pixel/project01/cruman/ModelData/PanArctic_0.5d_ERAINT_NOCTEM_RUN/Samples/PanArctic_0.5d_ERAINT_NOCTEM_RUN_198001

if __name__ == "__main__":
    main()
