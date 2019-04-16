import sys
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from os import listdir
from glob import glob
import cmocean

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

  percentage = open('percentage_seasonal.txt', 'w')
  percentage.write("Station Neg Pos Neg1 Neg2 Pos1 Pos2\n")

  # looping throught all the stations
  for lat, lon, name in zip(lats, lons, stnames):

    for season, sname in zip([[12, 1, 2],[6, 7, 8],[3, 4, 5],[9, 10, 11]], ['DJF','JJA','MAM','SON']):
      
      filepaths_n = []
      filepaths_p = []

      for month in season:
        print(name, month)
        
        for year in range(datai, dataf+1):

          # Open the .csv
          #filepaths_n.extend(glob('CSV/*{1}*_windpress_neg.csv'.format(month, year)))        
          filepaths_n.extend(glob('CSV/*{2}_{1}{0:02d}_windpress_neg.csv'.format(month, year, name)))
          filepaths_p.extend(glob('CSV/*{2}_{1}{0:02d}_windpress_pos.csv'.format(month, year, name)))
      #print(filepaths_n)
      #sys.exit()
              
      df_n = pd.concat((pd.read_csv(f, index_col=0) for f in filepaths_n), ignore_index=True)
      df_p = pd.concat((pd.read_csv(f, index_col=0) for f in filepaths_p), ignore_index=True)      

      #[10.0, 15.0, 20.0, 30.0, 50.0, 70.0, 100.0, 150.0, 200.0, 250.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0, 850.0, 900.0, 925.0, 950.0, 975.0, 1000.0]
      # Delete the upper levels of the atmosphere. I need only up to 700 hPa.
      df_n = df_n.drop(columns=['300.0', '400.0', '500.0', '600.0'])
      df_p = df_p.drop(columns=['300.0', '400.0', '500.0', '600.0'])

      p_neg = len(df_n.index)*100/(len(df_n.index) + len(df_p.index))
      p_pos = len(df_p.index)*100/(len(df_n.index) + len(df_p.index))

      centroids_n, histo_n, perc_n = kmeans_probability(df_n)

      cent = []
      histo = []
      perc = []
      shf = []

      if (perc_n[0] > perc_n[1]):
        k = 0
        j = 1
      else:
        k = 1
        j = 0

      cent.append(centroids_n[k])
      cent.append(centroids_n[j])

      histo.append(histo_n[k])
      histo.append(histo_n[j])

      perc.append(perc_n[k])
      perc.append(perc_n[j])

      shf.append('SHF-')
      shf.append('SHF-')

      #plot_wind(centroids_n[0], histo_n[0], perc_n[0], datai, dataf, name, "negative_type1", sname)
      #plot_wind(centroids_n[1], histo_n[1], perc_n[1], datai, dataf, name, "negative_type2", sname)      

      centroids_p, histo_p, perc_p = kmeans_probability(df_p)

      if (perc_p[0] > perc_p[1]):
        k = 0
        j = 1
      else:
        k = 1
        j = 0

      cent.append(centroids_p[k])
      cent.append(centroids_p[j])

      histo.append(histo_p[k])
      histo.append(histo_p[j])

      perc.append(perc_p[k])
      perc.append(perc_p[j])

      shf.append('SHF+')
      shf.append('SHF+')

      plot_wind_seasonal(cent, histo, perc, shf, datai, dataf, name, sname)

      #plot_wind(centroids_p[0], histo_p[0], perc_p[0], datai, dataf, name, "positive_type1", sname)
      #plot_wind(centroids_p[1], histo_p[1], perc_p[1], datai, dataf, name, "positive_type2", sname)

      percentage.write("{0} {1:2.2f} {2:2.2f} {3:2.2f} {4:2.2f} {5:2.2f} {6:2.2f} {7}\n".format(name, p_neg, p_pos, perc_n[0], perc_n[1], perc_p[0], perc_p[1], sname))

  percentage.close()
            
def plot_wind(centroids, histo, perc, datai, dataf, name, ptype, month):

  y = [700.0, 800.0, 850.0, 900.0, 925.0, 950.0, 975.0, 1000.0]
  x = np.arange(0,40,1)
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
  plt.xlim(0,40)
  plt.ylim(1000,700)
  plt.xticks(np.arange(0,40,5), fontsize=20)
  plt.yticks(y, fontsize=20)
  plt.title('{0}:{1} - {2:2.2f}% - {3} - {4}'.format(datai, dataf, perc, name, month), fontsize='20')
  plt.tight_layout()
  plt.savefig('Images/{0}_{1}{2}_{3}_{4}.png'.format(name, datai, dataf, ptype, month), pad_inches=0.0, bbox_inches='tight')
  plt.close()

  return None

def plot_wind_seasonal(centroids, histo, perc, shf, datai, dataf, name, period):

  y = [700.0, 800.0, 850.0, 900.0, 925.0, 950.0, 975.0, 1000.0]
  x = np.arange(0,40,1)
  X, Y= np.meshgrid(x, y)
  vmin=0
  vmax=15
  v = np.arange(vmin, vmax+1, 2)  

  fig = plt.figure(figsize=[28,16])

  for k, letter in zip(range(0,4), ['a', 'b', 'c', 'd']):
    subplt = '22{0}'.format(k+1)
    plt.subplot(subplt)

    CS = plt.contourf(X, Y, histo[k], v, cmap='cmo.haline', extend='max')
    CS.set_clim(vmin, vmax)
    plt.gca().invert_yaxis()
    plt.plot(centroids[k], y, color='white', marker='o', lw=4, markersize=10, markeredgecolor='k')
    CB = plt.colorbar(CS, extend='both', ticks=v)
    CB.ax.tick_params(labelsize=20)
    plt.xlim(0,39)
    plt.ylim(1000,700)
    plt.xticks(np.arange(0,40,5), fontsize=20)
    plt.yticks(y, fontsize=20)
    plt.title('({0}) {1:2.2f} % {2}'.format(letter, perc[k], shf[k]), fontsize='20')
  plt.tight_layout()
  plt.savefig('Images/{0}_{1}{2}_{3}.png'.format(name, datai, dataf, period), pad_inches=0.0, bbox_inches='tight')
  plt.close()

  return None

def kmeans_probability(df):
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

  # Dividing between the 2 clusters
  df_0 = df_a[labels,:]
  df_1 = df_a[~labels,:]

  # Getting the probability distribution. Bins of 0.5 m/s
  hist_0 = calc_histogram(df_0)
  hist_1 = calc_histogram(df_1)

  #print(np.mean(df_0, axis=0), np.mean(df_1, axis=0), kmeans.cluster_centers_)
  #sys.exit()

  return kmeans.cluster_centers_, [hist_0, hist_1], [df_0.shape[0]*100/df_a.shape[0], df_1.shape[0]*100/df_a.shape[0]]

def calc_histogram(df):

  hist_l = []
  bins = np.arange(0,40.25,1)
  for i in range(0, df.shape[1]):    
    hist, bins = np.histogram(df[:,i], bins=bins)
    hist_l.append(hist*100/sum(hist))

  return np.array(hist_l)
    




if __name__ == "__main__":
  main()
