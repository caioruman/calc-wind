import sys
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KernelDensity
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from os import listdir
from glob import glob
import cmocean

from common_functions import interpPressure, calc_height

'''
  - Read the wind data divided by positive and negative SHF values
  - Apply clustering analysis to each dataset to divide it between Unstable/Shear Driven and WSBL/VSBL

  to do: Weibull distribution, other things?
'''

def main():

  lats = []
  lons = []
  stnames = []

  main_folder = '/pixel/project01/cruman/ModelData/PanArctic_0.5d_ERAINT_NOCTEM_RUN/CSV_RCP'

  stations = open('stations.txt', 'r')
  for line in stations:
    aa = line.replace("\n", '').split(';')
    if (aa[0] != "#"):      
      lats.append(float(aa[3]))
      lons.append(float(aa[5]))
      stnames.append(aa[1].replace(',',"_"))

  datai = 1986
  dataf = 2015  

  # looping throught all the stations
  for lat, lon, name in zip(lats, lons, stnames):

    for season, sname in zip([[12, 1, 2],[6, 7, 8],[3, 4, 5],[9, 10, 11]], ['DJF','JJA','MAM','SON']):
      
      filepaths_n = []
      filepaths_p = []

      filepaths_tmp_n = []
      filepaths_tmp_p = []

      for month in season:
        print(name, month)
        
        for year in range(datai, dataf+1):

          # Open the .csv
          #filepaths_n.extend(glob('CSV/*{1}*_windpress_neg.csv'.format(month, year)))    
           
          filepaths_n.extend(glob('{3}/{1}/*{2}_{1}{0:02d}_windpress_neg.csv'.format(month, year, name, main_folder)))
          filepaths_p.extend(glob('{3}/{1}/*{2}_{1}{0:02d}_windpress_pos.csv'.format(month, year, name, main_folder)))

          #filepaths_n.extend(glob('CSV/*{2}_{1}{0:02d}_windpress_neg.csv'.format(month, year, name)))
          #filepaths_p.extend(glob('CSV/*{2}_{1}{0:02d}_windpress_pos.csv'.format(month, year, name)))

          filepaths_tmp_n.extend(glob('{3}/{1}/*{2}_{1}{0:02d}_neg.csv'.format(month, year, name, main_folder)))
          filepaths_tmp_p.extend(glob('{3}/{1}/*{2}_{1}{0:02d}_pos.csv'.format(month, year, name, main_folder)))
      #print(filepaths_n)
      #sys.exit()
              
      df_n = pd.concat((pd.read_csv(f, index_col=0) for f in filepaths_n), ignore_index=True)
      df_p = pd.concat((pd.read_csv(f, index_col=0) for f in filepaths_p), ignore_index=True)      

      df_tmp_n = pd.concat((pd.read_csv(f, index_col=0) for f in filepaths_tmp_n), ignore_index=True)
      df_tmp_p = pd.concat((pd.read_csv(f, index_col=0) for f in filepaths_tmp_p), ignore_index=True)

      #[10.0, 15.0, 20.0, 30.0, 50.0, 70.0, 100.0, 150.0, 200.0, 250.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0, 850.0, 900.0, 925.0, 950.0, 975.0, 1000.0]
      # Delete the upper levels of the atmosphere. I need only up to 700 hPa.
      df_n = df_n.drop(columns=['300.0', '400.0', '500.0', '600.0'])
      df_p = df_p.drop(columns=['300.0', '400.0', '500.0', '600.0'])
      
      df_tmp_n = df_tmp_n.drop(columns=['300.0', '400.0', '500.0', '600.0', 'SurfTemp', 'T2M', 'UV'])
      df_tmp_p = df_tmp_p.drop(columns=['300.0', '400.0', '500.0', '600.0', 'SurfTemp', 'T2M', 'UV'])         

      df_tmp_0, df_wind_0, df_tmp_1, df_wind_1 = kmeans_probability(df_n, df_tmp_n)      
           
      df_tmp_0_p, df_wind_0_p, df_tmp_1_p, df_wind_1_p = kmeans_probability(df_p, df_tmp_p)      

      

      x1, y1 = plot_scatter(df_tmp_0, df_wind_0, 'SHF-_type1_{0}.png'.format(name))
      x2, y2 = plot_scatter(df_tmp_1, df_wind_1, 'SHF-_type2_{0}.png'.format(name))

      plot_scatter2(x1, y1, x2, y2, 'SHF-_bothtypes_{0}.png'.format(name))

      #sys.exit()      

def plot_scatter(df_tmp, df_wind, fname):

  fig = plt.figure(figsize=[14,8])
  x = df_wind[:,-1]
  y = df_tmp[:,-2] - df_tmp[:,-1]
  y[y <= 0] = np.nan
  x[y <= 0] = np.nan
  plt.scatter(x, y)

  plt.ylabel('Temperature Difference')
  plt.xlabel('Wind')
  plt.savefig(fname)
  plt.close()
  
  return x, y

def plot_scatter2(x, y, x2, y2, fname):

  fig = plt.figure(figsize=[14,8])
  
  plt.scatter(x, y)
  plt.scatter(x2, y2)

  plt.ylabel('Temperature Difference')
  plt.xlabel('Wind')
  plt.savefig(fname)
  plt.close()
  
  return None


def kmeans_probability(df, df_tmp):
  '''
    For now fixed at 2 clusters

    returns: Array of the centroids, the two histograms and % of each group
  '''
  kmeans = KMeans(n_clusters=2, random_state=0).fit(df)
        
  # Getting the location of each group.
  pred = kmeans.predict(df)
  labels = np.equal(pred, 0)

  # Converting to numpy array
  df_a = np.array(df)
  df_tmp = np.array(df_tmp)

  # Dividing between the 2 clusters
  df_0 = df_a[labels,:]
  df_1 = df_a[~labels,:]

  df_tmp_0 = df_tmp[labels,:]
  df_tmp_1 = df_tmp[~labels,:]

  profileT_0 = np.mean(df_tmp_0, axis=0)
  profileT_1 = np.mean(df_tmp_1, axis=0)

  histT_0 = calc_histogram(df_tmp_0, -50, 20.1)
  histT_1 = calc_histogram(df_tmp_1, -50, 20.1)
  #wind_0 = df_0[:,6]
  #wind_1 = df_1[:,6]

  #deltaT_0 = df_tmp_0[:,10]-df_tmp_0[:,11]
  #deltaT_1 = df_tmp_1[:,10]-df_tmp_1[:,11]

  #plt.scatter(wind_0, deltaT_0)
  #plt.savefig('teste3.png')
  #plt.close()

  #plt.scatter(wind_1, deltaT_1)
  #plt.savefig('teste4.png')
  #sys.exit()

  # Getting the probability distribution. Bins of 0.5 m/s
  hist_0 = calc_histogram(df_0)
  hist_1 = calc_histogram(df_1)

  # Getting the probability distribution. Kernel Density  
  hist_0 = calc_kerneldensity(df_0)
  hist_1 = calc_kerneldensity(df_1)

  #print(np.mean(df_0, axis=0), np.mean(df_1, axis=0), kmeans.cluster_centers_)
  #sys.exit()

  return df_tmp_0, df_0, df_tmp_1, df_1
  #return kmeans.cluster_centers_, [hist_0, hist_1], [profileT_0, profileT_1], [histT_0, histT_1], [df_0.shape[0]*100/df_a.shape[0], df_1.shape[0]*100/df_a.shape[0]]

def calc_kerneldensity(df):
  hist_aux = []
  for i in range(0,8):
      kde_skl = KernelDensity(bandwidth=0.4)
      #aux = np.array(df_n['1000.0'])
      aux = np.copy(df[:,i])
      aux_grid2 = np.linspace(0,40,80)
      kde_skl.fit(aux[:, np.newaxis])
      log_pdf = kde_skl.score_samples(aux_grid2[:, np.newaxis])
      hist_aux.append(np.exp(log_pdf)*100)

  return hist_aux

def calc_histogram(df, irange=0, frange=40.25):

  hist_l = []
  bins = np.arange(irange,frange,1)
  for i in range(0, df.shape[1]):    
    hist, bins = np.histogram(df[:,i], bins=bins)
    hist_l.append(hist*100/sum(hist))

  return np.array(hist_l)
    




if __name__ == "__main__":
  main()
