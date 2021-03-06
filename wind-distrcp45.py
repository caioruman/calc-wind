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
args=parser.parse_args()

datai = args.anoi
dataf = args.anof


def main():

  exp = "PanArctic_0.5d_ERAINT_NOCTEM_RUN"
  exp = "PanArctic_0.5d_CanHisto_NOCTEM_RUN"
  exp = "PanArctic_0.5d_CanRCP45_NOCTEM_RUN"
#  exp = "cAYNWT_004deg_900x800_clef"
#  exp = "cPanCan011_675x540_SPN"
  main_folder = "/pixel/project01/cruman/ModelData/{0}".format(exp)
  folder = "/home/cruman/Scripts/calc-wind"
#  Cedar
  main_folder = "/home/cruman/projects/rrg-sushama-ab/teufel/{0}".format(exp)
#  Beluga
#  main_folder = "/home/poitras/projects/rrg-sushama-ab/poitras/storage_model/Output/DORVAL/{0}".format(exp)

#  datai = 1976
#  dataf = 2006

  # to be put in a loop later. 
  for year in range(datai, dataf+1):
    os.system('mkdir -p {1}/CSV_RCP45/{0}'.format(year, folder))

    for month in range(1,13):
  #year = 1980
  #month = 1

      # Sample point for testing. Try Ile St Madeleine later: 47.391348; -61.850658
      #sp_lat = 58.107914
      #sp_lon = -68.421492

    #  sp_lat = 47.391348
    #  sp_lon = -61.850658
      name = "71925__Cambridge_Bay__NT_YCB"
      if os.path.exists("{0}/CSV_RCP45/{4}/{1}_{2}{3:02d}_windpress_neg.csv".format(folder, name, year, month, year)):
        print("Month already calculated. skipping.")
        continue


      arq_dp = glob("{0}/Samples/{1}_{2}{3:02d}/dp*".format(main_folder, exp, year, month))[0]
      arq_dm = glob("{0}/Samples/{1}_{2}{3:02d}/dm*".format(main_folder, exp, year, month))[0]
      arq_pm = glob("{0}/Samples/{1}_{2}{3:02d}/pm*".format(main_folder, exp, year, month))[0]  

      # Reading SHF
      with RPN(arq_pm) as r:
        print("Opening file {0}".format(arq_pm))
#        sys.exit()
        shf = np.squeeze(r.variables["AH"][:])
        surf_temp = np.squeeze(r.variables["J8"][:]) - 273.15

        lons2d, lats2d = r.get_longitudes_and_latitudes_for_the_last_read_rec()  
      
      print(shf.shape, surf_temp.shape)

      # Reading 10m wind
      with RPN(arq_dm) as r:
        print("Opening file {0}".format(arq_dm))
        uu = np.squeeze(r.variables["UU"][:])
        vv = np.squeeze(r.variables["VV"][:])

        #t2m = np.squeeze(r.variables["TT"][:])  
        #t2m = r.get_first_record_for_name("TT", label="PAN_ERAI_DEF")

        #PAN_CAN85_CT
        #PAN_ERAI_DEF
        var = r.get_4d_field('TT', label="PAN_CAN45_CT")
        dates_tt = list(sorted(var.keys()))        
        key = [*var[dates_tt[0]].keys()][0]
        var_3d = np.asarray([var[d][key] for d in dates_tt])        
        t2m = var_3d.copy()
                
      uv = np.sqrt(np.power(uu, 2) + np.power(vv, 2))
      print(uu.shape, vv.shape, t2m.shape)  

      # Reading the wind on preassure levels

      with RPN(arq_dp) as r:
        print("Opening file {0}".format(arq_dp))
        uu_press = np.squeeze(r.variables["UU"][:])
        vv_press = np.squeeze(r.variables["VV"][:]) 

        tt_press = np.squeeze(r.variables["TT"][:]) 

        levels = [lev for lev in r.variables["UU"].sorted_levels]

        dates_d = np.array(r.variables["UU"].sorted_dates)

      print(tt_press.shape, uu_press.shape, vv_press.shape)

      uv_pressure = np.sqrt(np.power(uu_press, 2) + np.power(vv_press, 2))    
      print(levels)
      lats = []
      lons = []
      stnames = []

      stations = open('/home/cruman/scratch/Scripts/calc-wind/stations.txt', 'r')
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
        neg_shf = np.less_equal(shf[:, i, j], 0)

        neg_wind = uv[neg_shf, i, j]
        pos_wind = uv[~neg_shf, i, j]

        neg_t2m = t2m[neg_shf, i, j]
        pos_t2m = t2m[~neg_shf, i, j]

        neg_stemp = surf_temp[neg_shf, i, j]
        pos_stemp = surf_temp[~neg_shf, i, j]

        neg_wind_press = uv_pressure[neg_shf, 10:, i, j]
        pos_wind_press = uv_pressure[~neg_shf, 10:, i, j]

        neg_tt_press = tt_press[neg_shf, 10:, i, j]
        pos_tt_press = tt_press[~neg_shf, 10:, i, j]

        neg_dates_d = dates_d[neg_shf]
        pos_dates_d = dates_d[~neg_shf]

        df1 = pd.DataFrame(data=neg_wind_press, columns=levels[10:])
        df2 = pd.DataFrame(data=pos_wind_press, columns=levels[10:])

        df1 = df1.assign(Dates=neg_dates_d)
        df2 = df2.assign(Dates=pos_dates_d)

        df1.to_csv("{0}/CSV_RCP45/{4}/{1}_{2}{3:02d}_windpress_neg.csv".format(folder, name, year, month, year))
        df2.to_csv("{0}/CSV_RCP45/{4}/{1}_{2}{3:02d}_windpress_pos.csv".format(folder, name, year, month, year))

        df1 = pd.DataFrame(data=neg_tt_press, columns=levels[10:])
        df2 = pd.DataFrame(data=pos_tt_press, columns=levels[10:])

        df1 = df1.assign(SurfTemp=neg_stemp)
        df1 = df1.assign(T2M=neg_t2m)
        df1 = df1.assign(UV=neg_wind)

        df2 = df2.assign(SurfTemp=pos_stemp)
        df2 = df2.assign(T2M=pos_t2m)
        df2 = df2.assign(UV=pos_wind)

        df1 = df1.assign(Dates=neg_dates_d)
        df2 = df2.assign(Dates=pos_dates_d)

        df1.to_csv("{0}/CSV_RCP45/{4}/{1}_{2}{3:02d}_neg.csv".format(folder, name, year, month, year))
        df2.to_csv("{0}/CSV_RCP45/{4}/{1}_{2}{3:02d}_pos.csv".format(folder, name, year, month, year))
  #      sys.exit()



  # params = stats.exponweib.fit(data, floc=0, f0=1)
  # shape = params[1]
  # scale = params[3]

  # print('shape:',shape)
  # print('scale:',scale)

  # #### Plotting
  # fig = plt.figure(figsize=(11, 11), frameon=False, dpi=100)
  # # Histogram first
  # values,bins,hist = plt.hist(data,bins=51,range=(0,25),density=True)
  # center = (bins[:-1] + bins[1:]) / 2.

  # # Using all params and the stats function
  # plt.plot(center,stats.exponweib.pdf(center,*params),lw=4,label='scipy')

  #   #/pixel/project01/cruman/ModelData/PanArctic_0.5d_ERAINT_NOCTEM_RUN/Samples/PanArctic_0.5d_ERAINT_NOCTEM_RUN_198001
  # plt.legend()
  # plt.savefig('testea.png')
  # plt.close()

  # fig = plt.figure(figsize=(11, 11), frameon=False, dpi=100)

  # print(shape.data)
  # plt.plot(np.arange(len(data)), data)

  # plt.savefig("Teste2a.png")
  # plt.close()

  # fig = plt.figure(figsize=(11, 11), frameon=False, dpi=100)

  # print(shape.data)
  # plt.plot(np.arange(len(data)), shf[:,i,j])

  # plt.savefig("Teste_shfa.png")
  # plt.close()  




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
  main()
