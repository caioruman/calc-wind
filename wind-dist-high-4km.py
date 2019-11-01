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
parser.add_argument("exp", type=str, help="Ano", default=0)
args=parser.parse_args()

exp = args.exp
#print(exp)
#sys.exit()

def main(exp):
  
#  exp = "cAYNWT_004deg_900x800_clef"
#  exp = "cPanCan011_675x540_SPN"

  #exp2 = "PanCanada4km"
  #exp2 = "PanCanada10km"

  main_folder = "/pixel/project01/cruman/ModelData/{0}".format(exp)
  
  folder = "/home/cruman/Documents/Scripts/calc-wind"
#  Cedar
#  main_folder = "/home/cruman/projects/rrg-sushama-ab/teufel/{0}".format(exp)
#  Beluga
  main_folder = "/home/poitras/projects/rrg-sushama-ab/poitras/storage_model/Output/DORVAL/{0}".format(exp)
  #folder_nc = "/home/cruman/projects/rrg-sushama-ab/cruman/Simulations/{0}".format(exp2)

  datai = 1990
  dataf = 2010

  # to be put in a loop later. 
  for year in range(datai, dataf+1):
    os.system('mkdir -p CSV_RCP/{0}/{1}'.format(exp, year))

    for month in range(1,13):
  #year = 1980
  #month = 1

      # Sample point for testing. Try Ile St Madeleine later: 47.391348; -61.850658
      #sp_lat = 58.107914
      #sp_lon = -68.421492

    #  sp_lat = 47.391348
    #  sp_lon = -61.850658

      # I'll need to loop throught all the dm/pm/dp files, read them and concatenate in one array before processing
      arq_dp = sorted(glob("{0}/Samples/{1}_{2}{3:02d}/dp*".format(main_folder, exp, year, month)))
      arq_dm = sorted(glob("{0}/Samples/{1}_{2}{3:02d}/dm*".format(main_folder, exp, year, month)))
      arq_pm = sorted(glob("{0}/Samples/{1}_{2}{3:02d}/pm*".format(main_folder, exp, year, month))) 

      # Reading SHF
      ini = True
      for arq in arq_pm:

        with RPN(arq) as r:
          print("Opening file {0}".format(arq))
          shf = np.squeeze(r.variables["AHF"][:])
          shf = shf[:,4,:,:]
          #print(shf.shape)
          #sys.exit()
          #surf_temp = np.squeeze(r.variables["J8"][:]) - 273.15

          lons2d, lats2d = r.get_longitudes_and_latitudes_for_the_last_read_rec()  
          if ini:
              # Reading for the first time
              data_shf = shf
              #data_surf_temp = surf_temp
              ini = False
          else:
              # Stacking the arrays
              data_shf = np.vstack( (data_shf, shf) )
              #data_surf_temp = np.vstack( (data_surf_temp, surf_temp) )
      
      
#      print(shf.shape, surf_temp.shape)

      # Reading 10m wind
      ini = True
      for arq in arq_dm:
        with RPN(arq) as r:
          print("Opening file {0}".format(arq))
          uu = np.squeeze(r.variables["UU"][:])
          vv = np.squeeze(r.variables["VV"][:])

          t2m = np.squeeze(r.variables["TT"][:])  

          if ini:
            ini = False
            data_uu = uu
            data_vv = vv
            data_uv = np.sqrt( np.power(uu, 2) + np.power(vv, 2) )
            data_t2m = t2m
          else:
            data_uu = np.vstack( (data_uu, uu) )
            data_vv = np.vstack( (data_vv, vv) )
            data_uv = np.vstack( (data_uv, np.sqrt( np.power(uu, 2) + np.power(vv, 2) )) )
            data_t2m = np.vstack( (data_t2m, t2m) )

        #t2m = r.get_first_record_for_name("TT", label="PAN_ERAI_DEF")

        #PAN_CAN85_CT
        #PAN_ERAI_DEF
#       var = r.get_4d_field('TT', label="PAN_CAN85_CT")
#        dates_tt = list(sorted(var.keys()))        
#        key = [*var[dates_tt[0]].keys()][0]
#        var_3d = np.asarray([var[d][key] for d in dates_tt])        
#        t2m = var_3d.copy()
                
#        uv = np.sqrt(np.power(uu, 2) + np.power(vv, 2))
      #print(uu.shape, vv.shape, t2m.shape)  

      # Reading the wind on preassure levels
      ini = True
      for arq in arq_dp:

        with RPN(arq) as r:
          print("Opening file {0}".format(arq))
          
          uu_press = np.squeeze(r.variables["UU"][:])
          vv_press = np.squeeze(r.variables["VV"][:]) 

          tt_press = np.squeeze(r.variables["TT"][:]) 

          levels = [lev for lev in r.variables["UU"].sorted_levels]

          if ini:
            ini = False
            data_uu_press = uu_press
            data_vv_press = vv_press
            data_uv_press = np.sqrt(np.power(uu_press, 2) + np.power(vv_press, 2))
            data_tt_press = tt_press
          else:
            data_uu_press = np.vstack( (data_uu_press, uu_press) )
            data_vv_press = np.vstack( (data_vv_press, vv_press) )
            data_uv_press = np.vstack( (data_uv_press, np.sqrt(np.power(uu_press, 2) + np.power(vv_press, 2))) )
            data_tt_press = np.vstack( (data_tt_press, tt_press) )
      #print(data_uu_press.shape) # output: 248, 22, 635, 500
      #sys.exit()

#      print(tt_press.shape, uu_press.shape, vv_press.shape)

#      uv_pressure = np.sqrt(np.power(uu_press, 2) + np.power(vv_press, 2))    
#      print(levels)
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

        df1 = pd.DataFrame(data=neg_wind_press, columns=levels[10:])
        df2 = pd.DataFrame(data=pos_wind_press, columns=levels[10:])

        df1.to_csv("{0}/CSV_RCP/{5}/{4}/{1}_{2}{3:02d}_windpress_neg.csv".format(folder, name, year, month, year, exp))
        df2.to_csv("{0}/CSV_RCP/{5}/{4}/{1}_{2}{3:02d}_windpress_pos.csv".format(folder, name, year, month, year, exp))

        df1 = pd.DataFrame(data=neg_tt_press, columns=levels[10:])
        df2 = pd.DataFrame(data=pos_tt_press, columns=levels[10:])

        #df1 = df1.assign(SurfTemp=neg_stemp)
        df1 = df1.assign(T2M=neg_t2m)
        df1 = df1.assign(UV=neg_wind)

        #df2 = df2.assign(SurfTemp=pos_stemp)
        df2 = df2.assign(T2M=pos_t2m)
        df2 = df2.assign(UV=pos_wind)

        df1.to_csv("{0}/CSV_RCP/{5}/{4}/{1}_{2}{3:02d}_neg.csv".format(folder, name, year, month, year, exp))
        df2.to_csv("{0}/CSV_RCP/{5}/{4}/{1}_{2}{3:02d}_pos.csv".format(folder, name, year, month, year, exp))
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
