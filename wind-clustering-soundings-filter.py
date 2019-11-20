import sys
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KernelDensity
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from os import listdir
from glob import glob
import cmocean
import tarfile
import os
import re
import shutil

from common_functions import interpPressure, calc_height

'''
  - Read the soundings data
  - Read the model data
  - Attribute the value of SHF of the model data to the soundings data
  - Interpolate the soundings data to nice levels
  - Do the clustering analysis for the sounding data
'''

def main():

  lats = []
  lons = []
  stnames = []
  sheights = []

  stations = open('DatFiles/stations.txt', 'r')
  for line in stations:
    aa = line.replace("\n", '').split(';')
    if (aa[0] != "#"):      
      lats.append(float(aa[3]))
      lons.append(float(aa[5]))
      stnames.append(aa[1])
      sheights.append(float(aa[7]))

  datai = 1990
  dataf = 2010

  read_s = False

  #main_folder = '/pixel/project01/cruman/ModelData/PanArctic_0.5d_CanHisto_NOCTEM_RUN/CSV_RCP'
  main_folder = '/pixel/project01/cruman/ModelData/PanArctic_0.5d_ERAINT_NOCTEM_RUN/CSV_V2'
  main_folder = '/aos/home/cruman/Documents/Scripts/calc-wind/CSV/cPanCan011_675x540_SPN'

#  percentage = open('DatFiles/percentage_seasonal_v3_canesm2.txt', 'w')
#  percentage.write("Station Neg Pos Neg1 Neg2 Pos1 Pos2\n")

  # looping throught all the stations
  for lat, lon, name, sheight in zip(lats, lons, stnames, sheights):

    #for season, sname in zip([[12, 1, 2],[6, 7, 8],[3, 4, 5],[9, 10, 11]], ['DJF','JJA','MAM','SON']):
    for season, sname in zip([[12, 1, 2],[6, 7, 8]], ['DJF','JJA']):
      
      # Open the model data
      filepaths_n = []
      filepaths_p = []

      for month in season:
        print(name, month)
        
        for year in range(datai, dataf+1):
          
          reT = re.compile(r'.*?{0}_.*?{1}{2:02d}_.*?_windpress.*?'.format(name.replace(',',"_"), year, month))
          
          #os.mkdir('{0}/outdir'.format(main_folder))
          t = tarfile.open('{0}/{1}.tar.gz'.format(main_folder, year), 'r')
          t.extractall('{0}/outdir'.format(main_folder), members=[m for m in t.getmembers() if reT.search(m.name)])          
          
          #print os.listdir('outdir')
          # Open the .csv
          #filepaths_n.extend(glob('CSV/*{1}*_windpress_neg.csv'.format(month, year)))        
          filepaths_n.extend(glob('{3}/outdir/{1}/*{2}_{1}{0:02d}_*_windpress_neg.csv'.format(month, year, name.replace(',',"_"), main_folder)))
          filepaths_p.extend(glob('{3}/outdir/{1}/*{2}_{1}{0:02d}_*_windpress_pos.csv'.format(month, year, name.replace(',',"_"), main_folder)))      

          #print('{3}/outdir/{1}/*{2}_{1}{0:02d}_*_windpress_pos.csv'.format(month, year, name.replace(',',"_"), main_folder))
          #print(filepaths_n)
          #sys.exit()
              
      df_n = pd.concat((pd.read_csv(f, index_col=0) for f in filepaths_n), ignore_index=True)
      df_p = pd.concat((pd.read_csv(f, index_col=0) for f in filepaths_p), ignore_index=True)      

      #[10.0, 15.0, 20.0, 30.0, 50.0, 70.0, 100.0, 150.0, 200.0, 250.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0, 850.0, 900.0, 925.0, 950.0, 975.0, 1000.0]
      # Delete the upper levels of the atmosphere. I need only up to 700 hPa.
      df_n = df_n.drop(columns=['300.0', '400.0', '500.0', '600.0'])
      df_p = df_p.drop(columns=['300.0', '400.0', '500.0', '600.0'])

      # Open the soundings data
      # location: /pixel/project01/cruman/Data/Soundings/      
      # Data from the soundings
      #if read_s:
      df_height_p, df_height_n, df_temp_p, df_temp_n, df_wind_p, df_wind_n = filter_soundings(name, datai, dataf, sheight, df_n)
      
      df_height_n.to_csv('DatFiles/Soundings/{0}_{1}_{2}-{3}_height_neg.csv'.format(name.replace(',',"_"), sname, datai, dataf))
      df_height_p.to_csv('DatFiles/Soundings/{0}_{1}_{2}-{3}_height_pos.csv'.format(name.replace(',',"_"), sname, datai, dataf))
      df_temp_n.to_csv('DatFiles/Soundings/{0}_{1}_{2}-{3}_temp_neg.csv'.format(name.replace(',',"_"), sname, datai, dataf))
      df_temp_p.to_csv('DatFiles/Soundings/{0}_{1}_{2}-{3}_temp_pos.csv'.format(name.replace(',',"_"), sname, datai, dataf))
      df_wind_n.to_csv('DatFiles/Soundings/{0}_{1}_{2}-{3}_wind_neg.csv'.format(name.replace(',',"_"), sname, datai, dataf))
      df_wind_p.to_csv('DatFiles/Soundings/{0}_{1}_{2}-{3}_wind_pos.csv'.format(name.replace(',',"_"), sname, datai, dataf))

      



def filter_soundings(sname, datai, dataf, sheight, df_negative):
  '''
  its VERY VERY slow, because it loops throught the dates
  '''

  from datetime import datetime, timedelta
  # pressure levels
  #press_level =  [1000., 975., 950., 925., 900., 875., 850., 825., 800., 750., 700.]
  press_level =  [1000., 975., 950., 925., 900., 875., 850.]  
  press_level_columns =  [1000., 975., 950., 925., 900., 875., 850., 'Date']  

  # Read the soundings file
  f = '/pixel/project01/cruman/Data/Soundings/{0}/soundings_{1}_interp_v3.csv'.format(sname, sname[:5])
  #print(f)
  df = pd.read_csv(f, index_col=0) 

  # read the file, interpolate all the columns to the above pressure levels.
  dt = datetime(datai, 1, 2, 0, 0)
  date_f = datetime(dataf, 12, 31, 12, 0)    

  df_height = pd.DataFrame(columns=press_level_columns)
  df_temp = pd.DataFrame(columns=press_level_columns)
  df_wind = pd.DataFrame(columns=press_level_columns)

  df_height_n = pd.DataFrame(columns=press_level_columns)
  df_temp_n = pd.DataFrame(columns=press_level_columns)
  df_wind_n = pd.DataFrame(columns=press_level_columns)

  i = 0
  while dt <= date_f:

    i += 1
    if (i%500 == 0):
      print(dt, i)

    df_aux = df.query("Year == {0} and Month == {1} and Day == {2} and Hour == {3}".format(dt.year, dt.month, dt.day, dt.hour))

    df_aux = df_aux.drop(df_aux[df_aux.PRES > 1000].index)
    
    try:
      height = interpPressure(df_aux.PRES.values, press_level, df_aux.HGHT.values)-sheight
      #print(height+sheight)
      #print(df_aux.HGHT.values)
      temp = interpPressure(df_aux.PRES.values, press_level, df_aux.TEMP.values)
      wind = interpPressure(df_aux.PRES.values, press_level, df_aux.SKNT.values)/1.944
    except:
      dt = dt + timedelta(hours=12)
      continue

    height = height.tolist() + [dt]
    temp = temp.tolist() + [dt]
    wind = wind.tolist() + [dt]    

    df_aux2 = df_negative.query("Dates == '{0}'".format(dt))
    # Check this tomorrow after downloading the data from cedar
    #print(df_aux2.head())
    #print("Dates == '{0}'".format(dt))

    if (df_aux2.empty):
      #df_p.loc[len(df_p)] = row
      df_height.loc[len(df_height)] = height
      df_temp.loc[len(df_temp)] = temp
      df_wind.loc[len(df_wind)] = wind
    else:
      df_height_n.loc[len(df_height_n)] = height
      df_temp_n.loc[len(df_temp_n)] = temp
      df_wind_n.loc[len(df_wind_n)] = wind       
        
    dt = dt + timedelta(hours=12)  
  
  return df_height, df_height_n, df_temp, df_temp_n, df_wind, df_wind_n




if __name__ == "__main__":
  main()
