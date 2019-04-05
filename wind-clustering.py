import numpy as np
import pandas as pd
import sys
from scipy import stats
import matplotlib.pyplot as plt

from datetime import date, datetime, timedelta

import time

'''
  - Read the wind data divided by positive and negative SHF values
  - Apply clustering analysis to each dataset to divide it between Unstable/Shear Driven and WSBL/VSBL
'''

def main():

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

    # Open the .csv
    df1 = pd.read_csv("{0}_neg.csv".format(name), index_col=0)
    df2 = pd.read_csv("{0}_pos.csv".format(name), index_col=0)

    [10.0, 15.0, 20.0, 30.0, 50.0, 70.0, 100.0, 150.0, 200.0, 250.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0, 850.0, 900.0, 925.0, 950.0, 975.0, 1000.0]
    df1 = df1.drop(columns=['10.0', '15.0', '20.0', '30.0', '50.0', '70.0', '100.0', '150.0', '200.0', '250.0', '300.0', '400.0', '500.0', '600.0'])
    print(df1)
    sys.exit()

    # Delete the upper levels of the atmosphere. I need only up to 700 hPa.

    # For each level, do the K-means analysis
    




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
