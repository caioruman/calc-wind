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
import scipy.stats as s

'''
  - Read the wind data divided by positive and negative SHF values
  - PDF of the wind data. Weibull distribution.   
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

  #percentage = open('percentage_seasonal.txt', 'w')
  #percentage.write("Station Neg Pos Neg1 Neg2 Pos1 Pos2\n")

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

      df = pd.concat((pd.read_csv(f, index_col=0) for f in (filepaths_p+filepaths_n)), ignore_index=True)

      #[10.0, 15.0, 20.0, 30.0, 50.0, 70.0, 100.0, 150.0, 200.0, 250.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0, 850.0, 900.0, 925.0, 950.0, 975.0, 1000.0]
      # Delete the upper levels of the atmosphere. I need only up to 700 hPa.
      df_n = df_n.drop(columns=['300.0', '400.0', '500.0', '600.0'])
      df_p = df_p.drop(columns=['300.0', '400.0', '500.0', '600.0'])
      df = df.drop(columns=['300.0', '400.0', '500.0', '600.0'])

      p_neg = len(df_n.index)*100/(len(df_n.index) + len(df_p.index))
      p_pos = len(df_p.index)*100/(len(df_n.index) + len(df_p.index))

      centroids_n, histo_n, perc_n, wb_n, shape_n, scale_n = kmeans_probability(df_n)

      #weibull distribution
      wb_aux, shape, scale = calc_weibull(df)
      wb_aux_n, shape_n, scale_n = calc_weibull(df_n)
      wb_aux_p, shape_n, scale_p = calc_weibull(df_p)     

      plot_wind(wb_aux, datai, dataf, name, "alldata", sname)
      plot_wind(wb_aux_p, datai, dataf, name, "alldata_shf_plus", sname)
      plot_wind(wb_aux_n, datai, dataf, name, "alldata_shf_minus", sname)

      pdf = calc_kerneldensity(df)
      pdf_p = calc_kerneldensity(df_p)
      pdf_n = calc_kerneldensity(df_n)

      plot_wind(pdf, datai, dataf, name, "pdfalldata", sname)
      plot_wind(pdf_p, datai, dataf, name, "pdfalldata_shf_plus", sname)
      plot_wind(pdf_n, datai, dataf, name, "pdfalldata_shf_minus", sname)

      plot_difference(pdf-wb_aux, datai, dataf, name, "difference_alldata", sname)
      plot_difference(pdf_p-wb_aux_p, datai, dataf, name, "difference_alldata_shf_plus", sname)
      plot_difference(pdf_n-wb_aux_n, datai, dataf, name, "difference_alldata_shf_minus", sname)


      #plot_wind(centroids_p[0], histo_p[0], perc_p[0], datai, dataf, name, "positive_type1", sname)
      #plot_wind(centroids_p[1], histo_p[1], perc_p[1], datai, dataf, name, "positive_type2", sname)

      #percentage.write("{0} {1:2.2f} {2:2.2f} {3:2.2f} {4:2.2f} {5:2.2f} {6:2.2f} {7}\n".format(name, p_neg, p_pos, perc_n[0], perc_n[1], perc_p[0], perc_p[1], sname))

  #percentage.close()
            
def plot_wind(histo, datai, dataf, name, ptype, month):

  y = [700.0, 800.0, 850.0, 900.0, 925.0, 950.0, 975.0, 1000.0]
  x = np.arange(0,50,0.5)
  X, Y= np.meshgrid(x, y)
  vmin=0
  vmax=24
  v = np.arange(vmin, vmax+1, 2)

  fig = plt.figure(figsize=[14,8])
  #CS = plt.contourf(X, Y, histo, v, cmap='cmo.haline', extend='max')
  CS = plt.pcolormesh(X, Y, histo, cmap='cmo.haline')
  CS.set_clim(vmin, vmax)
  plt.gca().invert_yaxis()
  #plt.plot(centroids, y, color='white', marker='o', lw=4, markersize=10, markeredgecolor='k')
  CB = plt.colorbar(CS, extend='both', ticks=v)
  CB.ax.tick_params(labelsize=20)
  plt.xlim(0,40)
  plt.ylim(1000,700)
  plt.xticks(np.arange(0,40,5), fontsize=20)
  plt.yticks(y, fontsize=20)
  plt.title('{0}:{1} - {2} - {3}'.format(datai, dataf, name, month), fontsize='20')
  plt.tight_layout()
  plt.savefig('Images_WB/{0}_{1}{2}_{3}_{4}.png'.format(name, datai, dataf, ptype, month), pad_inches=0.0, bbox_inches='tight')
  plt.close()

  return None

def plot_difference(histo, datai, dataf, name, ptype, month):

  y = [700.0, 800.0, 850.0, 900.0, 925.0, 950.0, 975.0, 1000.0]
  x = np.arange(0,50,0.5)
  X, Y= np.meshgrid(x, y)
  vmin=-5
  vmax=5
  v = np.arange(vmin, vmax+1, -0.25)

  fig = plt.figure(figsize=[14,8])
  #CS = plt.contourf(X, Y, histo, v, cmap='cmo.haline', extend='max')
  CS = plt.pcolormesh(X, Y, histo, cmap='cmo.balance')
  CS.set_clim(vmin, vmax)
  plt.gca().invert_yaxis()
  #plt.plot(centroids, y, color='white', marker='o', lw=4, markersize=10, markeredgecolor='k')
  CB = plt.colorbar(CS, extend='both')
  CB.ax.tick_params(labelsize=20)
  plt.xlim(0,40)
  plt.ylim(1000,700)
  plt.xticks(np.arange(0,40,5), fontsize=20)
  plt.yticks(y, fontsize=20)
  plt.title('{0}:{1} - {2} - {3}'.format(datai, dataf, name, month), fontsize='20')
  #plt.tight_layout()
  plt.savefig('Images_WB/{0}_{1}{2}_{3}_{4}.png'.format(name, datai, dataf, ptype, month))#, pad_inches=0.0, bbox_inches='tight')
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

  wb_0, shape_0, scale_0 = calc_weibull(df_0)
  wb_1, shape_1, scale_1 = calc_weibull(df_1)

  wb = [wb_0, wb_1]
  hist = [hist_0, hist_1]
  shape = [shape_0, shape_1]
  scale = [scale_0, scale_1]
  #print(np.mean(df_0, axis=0), np.mean(df_1, axis=0), kmeans.cluster_centers_)
  #sys.exit()

  return kmeans.cluster_centers_, hist, [df_0.shape[0]*100/df_a.shape[0], df_1.shape[0]*100/df_a.shape[0]], wb, shape, scale

def calc_histogram(df):

  hist_l = []
  bins = np.arange(0,50.25,0.5)
  for i in range(0, df.shape[1]):    
    hist, bins = np.histogram(df[:,i], bins=bins)
    hist_l.append(hist*100/sum(hist))

  return np.array(hist_l)
    

def calc_kerneldensity(df):
  hist_aux = []
  df = np.array(df)
  for i in range(0,8):
      kde_skl = KernelDensity(bandwidth=0.4)
      #aux = np.array(df_n['1000.0'])
      aux = np.copy(df[:,i])
      aux_grid2 = np.linspace(0,50,100)
      kde_skl.fit(aux[:, np.newaxis])
      log_pdf = kde_skl.score_samples(aux_grid2[:, np.newaxis])
      hist_aux.append(np.exp(log_pdf)*100)

  return np.array(hist_aux)

def calc_weibull(df):
  wb_aux = []
  shape = []
  scale = []
  bins = np.arange(0,50.25,0.5)
  center = (bins[:-1] + bins[1:]) / 2.
  df = np.array(df)
  for i in range(0,8):
    params = s.exponweib.fit(df[:,i], floc=0, f0=1)
    shape.append(params[1])
    scale.append(params[3])
    wb_aux.append(s.exponweib.pdf(center,*params)*100)

  return np.array(wb_aux), shape, scale


if __name__ == "__main__":
  main()
