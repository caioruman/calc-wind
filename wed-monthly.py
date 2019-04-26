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
import argparse

from common_functions import calc_histogram, calc_kerneldensity, calc_height, interpPressure, calc_wind_density

'''
  - Read the wind data divided by positive and negative SHF values
  - Apply clustering analysis to each dataset to divide it between Unstable/Shear Driven and WSBL/VSBL
  - Calculate the Wind Energy Density for the lower levels
  - Interpolate to the 80m height and calculate the WED  
'''

def main():

  lats = []
  lons = []
  stnames = []

  parser = argparse.ArgumentParser()
  parser.add_argument("datai", help="Initial year of the calculations", type=int)
  parser.add_argument("dataf", help="Final year of the calculations", type=int)
  args = parser.parse_args()  

  main_folder = '/pixel/project01/cruman/ModelData/PanArctic_0.5d_CanHisto_NOCTEM_RUN/CSV_RCP'
  wheight = 80

  stations = open('stations.txt', 'r')
  for line in stations:
    aa = line.replace("\n", '').split(';')
    if (aa[0] != "#"):      
      lats.append(float(aa[3]))
      lons.append(float(aa[5]))
      stnames.append(aa[1].replace(',',"_"))

  datai = args.datai
  dataf = args.dataf

  R = 287.05 #Jkg-1K-1
  press = np.array([700.0, 800.0, 850.0, 900.0, 925.0, 950.0, 975.0, 1000.0])*100

  txtfile = open('wind_density_data-{0}-{1}_v2.dat'.format(datai,dataf), 'w')
  txtfile.write("station, month, type, perc_shf, perc_kmean, vmin, vmax, vmean, vstd, vrange, vskew, vkurtosis\n")

  # looping throught all the stations
  for lat, lon, name in zip(lats, lons, stnames):
    print(name)
    #for season, sname in zip([[12, 1, 2],[6, 7, 8],[3, 4, 5],[9, 10, 11]], ['DJF','JJA','MAM','SON']):
    for month in range(1,13):
      
      filepaths_n = []
      filepaths_p = []

      filepaths_tmp_n = []
      filepaths_tmp_p = []

      #for month in season:
      #  print(name, month)
        
      for year in range(datai, dataf+1):

        # Open the .csv
        #filepaths_n.extend(glob('CSV/*{1}*_windpress_neg.csv'.format(month, year)))        
        filepaths_n.extend(glob('{3}/{1}/*{2}_{1}{0:02d}_windpress_neg.csv'.format(month, year, name, main_folder)))
        filepaths_p.extend(glob('{3}/{1}/*{2}_{1}{0:02d}_windpress_pos.csv'.format(month, year, name, main_folder)))

        filepaths_tmp_n.extend(glob('{3}/{1}/*{2}_{1}{0:02d}_neg.csv'.format(month, year, name, main_folder)))
        filepaths_tmp_p.extend(glob('{3}/{1}/*{2}_{1}{0:02d}_pos.csv'.format(month, year, name, main_folder)))
      #print(filepaths_n)
      #print('{3}/{1}/*{2}_{1}{0:02d}_windpress_neg.csv'.format(month, year, name, main_folder))
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

      p_neg = len(df_n.index)*100/(len(df_n.index) + len(df_p.index))
      p_pos = len(df_p.index)*100/(len(df_n.index) + len(df_p.index))

      # Clustering analysis for Negative Values
      centroids_n, histo_n, perc_n, df_km_n, df_tmp_n = kmeans_probability(df_n, df_tmp_n)     

      perc = []
      w_data = []
      w_density = []
      
      perc.append(perc_n[0])
      perc.append(perc_n[1])      

      # Height of the model, returning the height (in meters) for the [700.0, 800.0, 850.0, 900.0, 925.0, 950.0, 975.0, 1000.0] array
      m_height = calc_height([month], datai, dataf)      

      #Interpolating the data to wheight (default: 80m)      
      w_data_0 = interpPressure(m_height, wheight, df_km_n[0])
      w_data_1 = interpPressure(m_height, wheight, df_km_n[1])

      pho_data_0 = interpPressure(m_height, wheight, press/(R*(df_tmp_n[0]+273.15)))
      pho_data_1 = interpPressure(m_height, wheight, press/(R*(df_tmp_n[1]+273.15)))

      #print(calc_vars(w_data_0))
      #print(calc_vars(w_data_1))

      # Wind Energy Density (W/m2) for the data      
      w_density.append(calc_wind_density(w_data_0, pho_data_0))
      w_density.append(calc_wind_density(w_data_1, pho_data_1))

      # Now doing the same thing to the SHF+ data

      centroids_p, histo_p, perc_p, df_km_p, df_tmp_p = kmeans_probability(df_p, df_tmp_p)

      perc.append(perc_p[0])
      perc.append(perc_p[1])    

      #Interpolating the data to wheight (default: 80m)      
      w_data_0 = interpPressure(m_height, wheight, df_km_p[0])
      w_data_1 = interpPressure(m_height, wheight, df_km_p[1])  

      pho_data_0 = interpPressure(m_height, wheight, press/(R*(df_tmp_p[0]+273.15)))
      pho_data_1 = interpPressure(m_height, wheight, press/(R*(df_tmp_p[1]+273.15)))  

      # Wind Energy Density (W/m2) for the data      
      w_density.append(calc_wind_density(w_data_0, pho_data_0))
      w_density.append(calc_wind_density(w_data_1, pho_data_1))

      #Calculate variables
      i = 0
      for item in w_density:
        vmin, vmax, vmean, vstd, vrange, vskew, vkurtosis = calc_vars(item)
        
        if (i < 2):
          aux = 'SHF-'
          aux_perc = p_neg
        else:
          aux = 'SHF+'
          aux_perc = p_pos

        # station, month, type, perc_shf, perc_kmean, vmin, vmax, vmean, vstd, vrange, vskew, vkurtosis
        txtfile.write("{0},{1},{2},{3:.2f},{4:.2f},{5:.2f},{6:.2f},{7:.2f},{8:.3f},{9:.2f},{10:.3f},{11:.3f}\n".format(name, month, aux, aux_perc, perc[i], vmin, vmax, vmean, vstd, vrange, vskew, vkurtosis))
        
        i += 1

      #percentage.write("{0} {1:2.2f} {2:2.2f} {3:2.2f} {4:2.2f} {5:2.2f} {6:2.2f} {7}\n".format(name, p_neg, p_pos, perc_n[0], perc_n[1], perc_p[0], perc_p[1], sname))
      #sys.exit()

  txtfile.close()

def calc_vars(data):
  from scipy.stats import kurtosis, skew

  vmin = np.min(data)
  vmax = np.max(data)
  vmean = np.mean(data)
  vrange = vmax - vmin  
  vstd = np.std(data)
  vskew = skew(data)
  vkurtosis = kurtosis(data)

  return vmin, vmax, vmean, vstd, vrange, vskew, vkurtosis
            

def plot_wind_seasonal(centroids, histo, perc, shf, datai, dataf, name, period):

  y = [700.0, 800.0, 850.0, 900.0, 925.0, 950.0, 975.0, 1000.0]
  #x = np.arange(0,40,1)
  x = np.arange(0,8000,10)
  X, Y= np.meshgrid(x, y)
  vmin=0
  vmax=1500
  v = np.arange(vmin, vmax+1, 15)  

  fig = plt.figure(figsize=[28,16])

  for k, letter in zip(range(0,4), ['a', 'b', 'c', 'd']):
    subplt = '22{0}'.format(k+1)
    plt.subplot(subplt)

    CS = plt.contourf(X, Y, histo[k], cmap='cmo.haline', extend='max')
    #CS.set_clim(vmin, vmax)
    plt.gca().invert_yaxis()
    plt.plot(centroids[k], y, color='white', marker='o', lw=4, markersize=10, markeredgecolor='k')
    if (k % 2):
      CB = plt.colorbar(CS, extend='both', ticks=v)
      CB.ax.tick_params(labelsize=20)
    #plt.xlim(0,800)
    plt.ylim(1000,700)
    #plt.xticks(np.arange(0,40,5), fontsize=20)
    plt.yticks(y, fontsize=20)
    plt.title('({0}) {1:2.2f} % {2}'.format(letter, perc[k], shf[k]), fontsize='20')
  plt.tight_layout()
  plt.savefig('Images_WD/{0}_{1}{2}_{3}_wpe.png'.format(name, datai, dataf, period), pad_inches=0.0, bbox_inches='tight')
  plt.close()
  sys.exit()

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
  df_b = np.array(df_tmp)

  # Dividing between the 2 clusters
  df_0 = df_a[labels,:]
  df_1 = df_a[~labels,:]

  df_tmp_0 = df_b[labels,:]
  df_tmp_1 = df_b[~labels,:]

  # Getting the probability distribution. Bins of 0.5 m/s
  hist_0 = calc_histogram(df_0)
  hist_1 = calc_histogram(df_1)

  # Getting the probability distribution. Kernel Density  
  hist_0 = calc_kerneldensity(df_0)
  hist_1 = calc_kerneldensity(df_1)

  #print(np.mean(df_0, axis=0), np.mean(df_1, axis=0), kmeans.cluster_centers_)
  #sys.exit()

  return kmeans.cluster_centers_, [hist_0, hist_1], [df_0.shape[0]*100/df_a.shape[0], df_1.shape[0]*100/df_a.shape[0]], [df_0, df_1], [df_tmp_0, df_tmp_1]


    




if __name__ == "__main__":
  main()
