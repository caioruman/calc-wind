import sys
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from os import listdir
from glob import glob
import cmocean
import hdbscan

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

  datai = 1986
  dataf = 2015

  percentage = open('percentage_hdb.txt', 'w')
  percentage.write("Station Neg Pos Neg1 Neg2 Pos1 Pos2\n")

  # looping throught all the stations
  for lat, lon, name in zip(lats, lons, stnames):

    for month in range(1,13):

      print(name, month)
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

      p_neg = len(df_n.index)*100/(len(df_n.index) + len(df_p.index))
      p_pos = len(df_p.index)*100/(len(df_n.index) + len(df_p.index))

      centroids_n, histo_n, perc_n = hdb_probability(df_n)

      plot_wind(centroids_n[0], histo_n[0], perc_n[0], datai, dataf, name, "negative_type1", month)
      plot_wind(centroids_n[1], histo_n[1], perc_n[1], datai, dataf, name, "negative_type2", month)

      centroids_p, histo_p, perc_p = hdb_probability(df_p)

      plot_wind(centroids_p[0], histo_p[0], perc_p[0], datai, dataf, name, "positive_type1", month)
      plot_wind(centroids_p[1], histo_p[1], perc_p[1], datai, dataf, name, "positive_type2", month)

      percentage.write("{0} {7} {1:2.2f} {2:2.2f} {3:2.2f} {4:2.2f} {5:2.2f} {6:2.2f}\n".format(name, p_neg, p_pos, perc_n[0], perc_n[1], perc_p[0], perc_p[1], month))
      sys.exit()
  percentage.close()
            
def plot_wind(centroids, histo, perc, datai, dataf, name, ptype, month):

  y = [700.0, 800.0, 850.0, 900.0, 925.0, 950.0, 975.0, 1000.0]
  x = np.arange(0,80,1)
  X, Y= np.meshgrid(x, y)
  vmin=0
  vmax=15
  v = np.arange(vmin, vmax+1, 2)

  fig = plt.figure(figsize=[14,8])
  CS = plt.contourf(X, Y, histo, v, cmap='cmo.haline', extend='max')
  CS.set_clim(vmin, vmax)
  plt.gca().invert_yaxis()
  plt.plot(centroids, y, color='white', marker='o', lw=4, markersize=10, markeredgecolor='k')
  CB = plt.colorbar(CS, extend='both', ticks=v)
  CB.ax.tick_params(labelsize=20)
  plt.xlim(0,75)
  plt.ylim(1000,700)
  plt.xticks(np.arange(0,80,5), fontsize=20)
  plt.yticks(y, fontsize=20)
  plt.title('{0}:{1} - {2:2.2f}% - {3}'.format(datai, dataf, perc, name), fontsize='20')
  plt.savefig('Images_HDB/{0}_{1}{2}_{3}_{4}_gm.png'.format(name, datai, dataf, ptype, month))
  plt.close()

  return None

def hdb_probability(df):
  '''
    For now fixed at 2 clusters

    returns: Array of the centroids, the two histograms and % of each group

    Now using HDBscan with min cluster size 9
  '''
  #kmeans = KMeans(n_clusters=2, random_state=0).fit(df)
  #gm = GaussianMixture(n_components=2).fit(df)
  hdb = hdbscan.HDBSCAN(min_cluster_size=1000)
  hdb_pred = hdb.fit(df)
        
  # Getting the location of each group.  
  # Converting to numpy array
  df_a = np.array(df)
  #print(df_a.shape)

  #Removing the noise
  labels = np.equal(hdb_pred.labels_, -1)
  df_a = df_a[~labels,:]

  labels = np.equal(hdb_pred.labels_, 0)
  print(np.unique(hdb_pred.labels_, return_counts=True))

  # Dividing between the 2 clusters
  df_0 = df_a[labels,:]
  df_1 = df_a[~labels,:]
  #print(df_0.shape, df_1.shape)


  # Getting the probability distribution. Bins of 0.5 m/s
  hist_0 = calc_histogram(df_0)
  hist_1 = calc_histogram(df_1)
  #print([df_0.shape[0]*100/df_a.shape[0], df_1.shape[0]*100/df_a.shape[0]])

  return [df_0.mean(axis=0), df_1.mean(axis=0)], [hist_0, hist_1], [df_0.shape[0]*100/df_a.shape[0], df_1.shape[0]*100/df_a.shape[0]]

def calc_histogram(df):

  hist_l = []
  bins = np.arange(0,80.5,1)
  for i in range(0, df.shape[1]):    
    hist, bins = np.histogram(df[:,i], bins=bins)
    hist_l.append(hist*100/sum(hist))

  return np.array(hist_l)
    




if __name__ == "__main__":
  main()
