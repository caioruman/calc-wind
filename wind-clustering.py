import sys
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from os import listdir
from glob import glob

'''
  - Read the wind data divided by positive and negative SHF values
  - Apply clustering analysis to each dataset to divide it between Unstable/Shear Driven and WSBL/VSBL

  to do: Weibull distribution, other things?
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

  datai = 1980
  dataf = 2015

  # looping throught all the stations
  for lat, lon, name in zip(lats, lons, stnames):

    for month in range(1,13):

      filepaths_n = []
      filepaths_p = []
      for year in range(datai, dataf+1):

        # Open the .csv
        #filepaths_n.extend(glob('CSV/*{1}*_windpress_neg.csv'.format(month, year)))        
        filepaths_n.extend(glob('CSV/*{1}{0:02d}_windpress_neg.csv'.format(month, year)))
        filepaths_p.extend(glob('CSV/*{1}{0:02d}_windpress_pos.csv'.format(month, year)))
              
      df_n = pd.concat((pd.read_csv(f, index_col=0) for f in filepaths_n), ignore_index=True)
      df_p = pd.concat((pd.read_csv(f, index_col=0) for f in filepaths_p), ignore_index=True)      

      #[10.0, 15.0, 20.0, 30.0, 50.0, 70.0, 100.0, 150.0, 200.0, 250.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0, 850.0, 900.0, 925.0, 950.0, 975.0, 1000.0]
      # Delete the upper levels of the atmosphere. I need only up to 700 hPa.
      df_n = df_n.drop(columns=['300.0', '400.0', '500.0', '600.0'])
      df_p = df_p.drop(columns=['300.0', '400.0', '500.0', '600.0'])

      centroids_n, histo_n, perc_n = kmeans_probability(df_n)
      centroids_p, histo_p, perc_p = kmeans_probability(df_p)

      # do stuff


def kmeans_probability(df):
  '''
    For now fixed at 2 clusters

    returns: Array of the centroids, the two histograms and % of each group
  '''

  kmeans = KMeans(n_clusters=2, random_state=0).fit(df)
        
  # Getting the location of each group.
  pred = kmeans_n.predict(df)
  labels = np.equal(pred, 0)

  # Converting to numpy array
  df_a = np.array(df)

  # Dividing between the 2 clusters
  df_0 = df_a[labels,:]
  df_1 = df_a[~labels,:]

  # Getting the probability distribution. Bins of 0.5 m/s
  hist_0 = calc_histogram(df_0)
  hist_1 = calc_histogram(df_1)

  return kmeans_n.cluster_centers_, [hist_0, hist_1], [df_0.shape[0]*100/df_a.shape[0], df_1.shape[0]*100/df_a.shape[0]]

def calc_histogram(df):

  hist_l = []
  bins = np.arange(0,80.5,0.5)
  for i in range(0, df.columns):
      hist, bins = np.histogram(df[:,i], bins=bins)
      hist_l.append(hist*100/sum(hist))

  return np.array(hist_l)
    




if __name__ == "__main__":
  main()
