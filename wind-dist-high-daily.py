import numpy as np
import pandas as pd
import sys
from scipy import stats
import matplotlib.pyplot as plt
import os
import argparse

from datetime import date, datetime, timedelta

from glob import glob
from rpn.rpn import RPN
from rpn.domains.rotated_lat_lon import RotatedLatLon
from rpn import level_kinds

from netCDF4 import Dataset
import time

'''
  - Calculates the probability distribution for the wind at the surface and pressure levels
  - Calculates the weibull distribution and return the parameters

  - Open the RPN files for each month
  - Group the data in one array
  - Get the timeseries for each station
  - Divide the timeseries between SHF > 0 and SHF < 0
  - K-mean clustering of those two timeseries
'''

parser=argparse.ArgumentParser(description='Separates the wind profiles based on SHF', formatter_class=argparse.RawTextHelpFormatter)
#parser.add_argument("-op", "--opt-arg", type=str, dest='opcional', help="Algum argumento opcional no programa", default=False)
parser.add_argument("anoi", type=int, help="Ano", default=0)
parser.add_argument("anof", type=int, help="Anof", default=0)
parser.add_argument("exp", type=str, help="exp", default=0)
args=parser.parse_args()

datai = args.anoi
dataf = args.anof
exp = args.exp
#print(exp)
#sys.exit()
#exp = "cAYNWT_004deg_900x800_clef"
#exp = "cPanCan011_675x540_SPN"

def main(exp):
  
  #exp = "cAYNWT_004deg_900x800_clef"
  #exp = "cPanCan011_675x540_SPN"

  #exp2 = "PanCanada4km"
  #exp2 = "PanCanada10km"

  main_folder = "/pixel/project01/cruman/ModelData/{0}".format(exp)
  
  folder = "/home/cruman/Documents/Scripts/calc-wind"
#  folder = "/home/cruman/projects/rrg-sushama-ab/cruman/Data"
  folder = "/lustre03/project/6004670/cruman/Data"
#  Cedar
#  main_folder = "/home/cruman/projects/rrg-sushama-ab/teufel/{0}".format(exp)
#  Beluga
  main_folder = "/home/poitras/projects/rrg-sushama-ab/poitras/storage_model/Output/DORVAL/{0}".format(exp)
  #folder_nc = "/home/cruman/projects/rrg-sushama-ab/cruman/Simulations/{0}".format(exp2)

#  datai = 1991
#  dataf = 2010
#  print(datai, dataf)
#  sys.exit()

  # to be put in a loop later. 
  for year in range(datai, dataf+1):
    os.system('mkdir -p {2}/CSV/{0}/{1}'.format(exp, year, folder))

    for month in range(1,13):
  #year = 1980
  #month = 1

      # Sample point for testing. Try Ile St Madeleine later: 47.391348; -61.850658
      #sp_lat = 58.107914
      #sp_lon = -68.421492

      # Check if the file exists. if yes, jump to the next month      
      # first station on the station.txt file: 71925__Cambridge_Bay__NT_YCB_199101_windpress_pos.csv
      name = "71925__Cambridge_Bay__NT_YCB"
      if os.path.exists("{0}/CSV/{5}/{4}/{1}_{2}{3:02d}_windpress_neg.csv".format(folder, name, year, month, year, exp)):
        print("Month already calculated. skipping.")
        continue

    #  sp_lat = 47.391348
    #  sp_lon = -61.850658
      print(year, month, " ")      

      #print("left the loop")
      #sys.exit()
      # I'll need to loop throught all the dm/pm/dp files, read them and concatenate in one array before processing
      arq_dp = sorted(glob("{0}/Samples/{1}_{2}{3:02d}/dp*".format(main_folder, exp, year, month)))
      arq_dm = sorted(glob("{0}/Samples/{1}_{2}{3:02d}/dm*".format(main_folder, exp, year, month)))
      arq_pm = sorted(glob("{0}/Samples/{1}_{2}{3:02d}/pm*".format(main_folder, exp, year, month))) 

      # Reading SHF
      ini = True
      mm = 0
      for arqpm, arqdm, arqdp in zip(arq_pm, arq_dm, arq_dp):
        mm += 1
        print(mm)

        # Check if the file exists. if yes, jump to the next month      
        # first station on the station.txt file: 71925__Cambridge_Bay__NT_YCB_199101_windpress_pos.csv
        name = "71925__Cambridge_Bay__NT_YCB"
        if os.path.exists("{0}/CSV/{5}/{4}/{1}_{2}{3:02d}_{6:02d}_windpress_neg.csv".format(folder, name, year, month, year, exp, mm)):
          print("Day already calculated. skipping.")
          print(year, month, mm)
          continue

        with RPN(arqpm) as r:
          print("Opening file {0}".format(arqpm))
          if exp == "cPanCan011_675x540_SPN":
            shf = np.squeeze(r.variables["AH"][:])
          else:
            shf = np.squeeze(r.variables["AHF"][:])
            shf = shf[:,4,:,:]
          #print(shf.shape)
          #sys.exit()
          #surf_temp = np.squeeze(r.variables["J8"][:]) - 273.15

          
          if ini:
            # Reading for the first time              
            lons2d, lats2d = r.get_longitudes_and_latitudes_for_the_last_read_rec()  
            #data_surf_temp = surf_temp
            ini = False
          data_shf = shf
          #else:
              # Stacking the arrays
              #data_shf = np.vstack( (data_shf, shf) )
              #data_surf_temp = np.vstack( (data_surf_temp, surf_temp) )

      # Reading 10m wind
       # ini = True        
        with RPN(arqdm) as r:
          print("Opening file {0}".format(arqdm))
          uu = np.squeeze(r.variables["UU"][:])
          vv = np.squeeze(r.variables["VV"][:])

          t2m = np.squeeze(r.variables["TT"][:])  

          
          data_uu = uu
          data_vv = vv
          data_uv = np.sqrt( np.power(uu, 2) + np.power(vv, 2) )
          data_t2m = t2m                  

      # Reading the wind on preassure levels
      
        with RPN(arqdp) as r:
          print("Opening file {0}".format(arqdp))
          
          uu_press = np.squeeze(r.variables["UU"][:])
          vv_press = np.squeeze(r.variables["VV"][:]) 

          tt_press = np.squeeze(r.variables["TT"][:]) 

          levels = [lev for lev in r.variables["UU"].sorted_levels]
        
          data_uu_press = uu_press
          data_vv_press = vv_press
          data_uv_press = np.sqrt(np.power(uu_press, 2) + np.power(vv_press, 2))
          data_tt_press = tt_press

          dates_d = np.array(r.variables["UU"].sorted_dates)
          
        lats = []
        lons = []
        stnames = []

        stations = open('stations.txt', 'r')
        for line in stations:
          aa = line.replace("\n", '').split(';')
          if (aa[0] != "#"):      
            lats.append(float(aa[3]))
            lons.append(float(aa[5]))
            stnames.append(aa[1].replace(',',"_"))

        # looping throught all the stations
        for lat, lon, name in zip(lats, lons, stnames):

          # Extract the info from the grid 
          i, j = geo_idx([lat, lon], np.array([lats2d, lons2d]))

          # Separating the negative and positive values, to apply to the wind
          neg_shf = np.less_equal(data_shf[:, i, j], 0)

          neg_wind = data_uv[neg_shf, i, j]
          pos_wind = data_uv[~neg_shf, i, j]

          neg_t2m = data_t2m[neg_shf, i, j]
          pos_t2m = data_t2m[~neg_shf, i, j]

          #neg_stemp = data_surf_temp[neg_shf, i, j]
          #pos_stemp = data_surf_temp[~neg_shf, i, j]

          neg_wind_press = data_uv_press[neg_shf, 10:, i, j]
          pos_wind_press = data_uv_press[~neg_shf, 10:, i, j]

          neg_tt_press = data_tt_press[neg_shf, 10:, i, j]
          pos_tt_press = data_tt_press[~neg_shf, 10:, i, j]

          neg_dates_d = dates_d[neg_shf]
          pos_dates_d = dates_d[~neg_shf]

          df1 = pd.DataFrame(data=neg_wind_press, columns=levels[10:])
          df2 = pd.DataFrame(data=pos_wind_press, columns=levels[10:])    

          df1 = df1.assign(Dates=neg_dates_d)
          df2 = df2.assign(Dates=pos_dates_d)      

          print("{0}/CSV/{5}/{4}/{1}_{2}{3:02d}_{6:02d}_windpress_neg.csv".format(folder, name, year, month, year, exp, mm))
          df1.to_csv("{0}/CSV/{5}/{4}/{1}_{2}{3:02d}_{6:02d}_windpress_neg.csv".format(folder, name, year, month, year, exp, mm))
          df2.to_csv("{0}/CSV/{5}/{4}/{1}_{2}{3:02d}_{6:02d}_windpress_pos.csv".format(folder, name, year, month, year, exp, mm))

          df1 = pd.DataFrame(data=neg_tt_press, columns=levels[10:])
          df2 = pd.DataFrame(data=pos_tt_press, columns=levels[10:])

          #df1 = df1.assign(SurfTemp=neg_stemp)
          df1 = df1.assign(T2M=neg_t2m)
          df1 = df1.assign(UV=neg_wind)

          #df2 = df2.assign(SurfTemp=pos_stemp)
          df2 = df2.assign(T2M=pos_t2m)
          df2 = df2.assign(UV=pos_wind)

          df1 = df1.assign(Dates=neg_dates_d)
          df2 = df2.assign(Dates=pos_dates_d)

          df1.to_csv("{0}/CSV/{5}/{4}/{1}_{2}{3:02d}_{6:02d}_neg.csv".format(folder, name, year, month, year, exp, mm))
          df2.to_csv("{0}/CSV/{5}/{4}/{1}_{2}{3:02d}_{6:02d}_pos.csv".format(folder, name, year, month, year, exp, mm))
  #      sys.exit()


def geo_idx(dd, dd_array, type="lat"):
  '''
    search for nearest decimal degree in an array of decimal degrees and return the index.
    np.argmin returns the indices of minium value along an axis.
    so subtract dd from all values in dd_array, take absolute value and find index of minimum.
    
    Differentiate between 2-D and 1-D lat/lon arrays.
    for 2-D arrays, should receive values in this format: dd=[lat, lon], dd_array=[lats2d,lons2d]
  '''
  if type == "lon" and len(dd_array.shape) == 1:
    dd_array = np.where(dd_array <= 180, dd_array, dd_array - 360)

  if (len(dd_array.shape) < 2):
    geo_idx = (np.abs(dd_array - dd)).argmin()
  else:
    if (dd_array[1] < 0).any():
      dd_array[1] = np.where(dd_array[1] <= 180, dd_array[1], dd_array[1] - 360)

    a = abs( dd_array[0]-dd[0] ) + abs(  np.where(dd_array[1] <= 180, dd_array[1], dd_array[1] - 360) - dd[1] )
    i,j = np.unravel_index(a.argmin(), a.shape)
    geo_idx = [i,j]

  return geo_idx


if __name__ == "__main__":
  main(exp)
