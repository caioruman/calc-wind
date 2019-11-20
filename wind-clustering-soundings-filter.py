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
          #sys.exit()
          #print("{2}_{1}{0:02d}_windpress".format(month, year, name.replace(',',"_")))
          #for member in t.getmembers():            
            #if "{2}_{1}{0:02d}_*_windpress".format(month, year, name.replace(',',"_")) in member.name:
              #t.extract(member, '{0}/outdir'.format(main_folder))

          #print os.listdir('outdir')
          # Open the .csv
          #filepaths_n.extend(glob('CSV/*{1}*_windpress_neg.csv'.format(month, year)))        
          filepaths_n.extend(glob('{3}outdir/{1}/*{2}_{1}{0:02d}_*_windpress_neg.csv'.format(month, year, name.replace(',',"_"), main_folder)))
          filepaths_p.extend(glob('{3}outdir/{1}/*{2}_{1}{0:02d}_*_windpress_pos.csv'.format(month, year, name.replace(',',"_"), main_folder)))      

          print(filepaths_n)
          sys.exit()
              
      df_n = pd.concat((pd.read_csv(f, index_col=0) for f in filepaths_n), ignore_index=True)
      df_p = pd.concat((pd.read_csv(f, index_col=0) for f in filepaths_p), ignore_index=True)      

      #[10.0, 15.0, 20.0, 30.0, 50.0, 70.0, 100.0, 150.0, 200.0, 250.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0, 850.0, 900.0, 925.0, 950.0, 975.0, 1000.0]
      # Delete the upper levels of the atmosphere. I need only up to 700 hPa.
      df_n = df_n.drop(columns=['300.0', '400.0', '500.0', '600.0'])
      df_p = df_p.drop(columns=['300.0', '400.0', '500.0', '600.0'])

      # Open the soundings data
      # location: /pixel/project01/cruman/Data/Soundings/      
      # Data from the soundings
      if read_s:
        df_height_p, df_height_n, df_temp_p, df_temp_n, df_wind_p, df_wind_n = filter_soundings(name, datai, dataf, sheight, df_n)
        
        df_height_n.to_csv('DatFiles/Soundings/{0}_{1}_{2}-{3}_height_neg.csv'.format(name.replace(',',"_"), sname, datai, dataf))
        df_height_p.to_csv('DatFiles/Soundings/{0}_{1}_{2}-{3}_height_pos.csv'.format(name.replace(',',"_"), sname, datai, dataf))
        df_temp_n.to_csv('DatFiles/Soundings/{0}_{1}_{2}-{3}_temp_neg.csv'.format(name.replace(',',"_"), sname, datai, dataf))
        df_temp_p.to_csv('DatFiles/Soundings/{0}_{1}_{2}-{3}_temp_pos.csv'.format(name.replace(',',"_"), sname, datai, dataf))
        df_wind_n.to_csv('DatFiles/Soundings/{0}_{1}_{2}-{3}_wind_neg.csv'.format(name.replace(',',"_"), sname, datai, dataf))
        df_wind_p.to_csv('DatFiles/Soundings/{0}_{1}_{2}-{3}_wind_pos.csv'.format(name.replace(',',"_"), sname, datai, dataf))

        continue
      else:
        df_height_p = pd.read_csv('DatFiles/Soundings/{0}_{1}_{2}-{3}_height_neg.csv'.format(name.replace(',',"_"), sname, datai, dataf), index_col=0)
        df_height_n = pd.read_csv('DatFiles/Soundings/{0}_{1}_{2}-{3}_height_pos.csv'.format(name.replace(',',"_"), sname, datai, dataf), index_col=0)
        df_temp_p = pd.read_csv('DatFiles/Soundings/{0}_{1}_{2}-{3}_temp_neg.csv'.format(name.replace(',',"_"), sname, datai, dataf), index_col=0)
        df_temp_n = pd.read_csv('DatFiles/Soundings/{0}_{1}_{2}-{3}_temp_pos.csv'.format(name.replace(',',"_"), sname, datai, dataf), index_col=0)
        df_wind_p = pd.read_csv('DatFiles/Soundings/{0}_{1}_{2}-{3}_wind_neg.csv'.format(name.replace(',',"_"), sname, datai, dataf), index_col=0)
        df_wind_n = pd.read_csv('DatFiles/Soundings/{0}_{1}_{2}-{3}_wind_pos.csv'.format(name.replace(',',"_"), sname, datai, dataf), index_col=0)
      
      sys.exit()
      #################
      

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

      plot_wind_seasonal(cent, histo, perc, shf, datai, dataf, name.replace(',',"_"), sname, season)

      #plot_wind(centroids_p[0], histo_p[0], perc_p[0], datai, dataf, name, "positive_type1", sname)
      #plot_wind(centroids_p[1], histo_p[1], perc_p[1], datai, dataf, name, "positive_type2", sname)

#     percentage.write("{0} {1:2.2f} {2:2.2f} {3:2.2f} {4:2.2f} {5:2.2f} {6:2.2f} {7}\n".format(name, p_neg, p_pos, perc_n[0], perc_n[1], perc_p[0], perc_p[1], sname))
      #sys.exit()

#  percentage.close()

def separate_dataframes(df, df_negative):

  # loop throught the df, check the date with the date in df_negative
  df_n = pd.DataFrame(columns=df.columns)
  df_p = pd.DataFrame(columns=df.columns)

  print(df.head())
  print(df_negative.head())
  print(df_n.head())
  for index, row in df.iterrows():    

    df_aux = df_negative.query("Date == {0}".format(row.Date))

    if (df_aux.empty):
      df_p.loc[len(df_p)] = row
    else:
      df_n.loc[len(df_n)] = row
    
    print(index)
    print(row)
    sys.exit()  

  return df_n, df_p

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
            
def plot_wind_seasonal(centroids, histo, perc, shf, datai, dataf, name, period, season):

  y = [700.0, 800.0, 850.0, 900.0, 925.0, 950.0, 975.0, 1000.0]
  y = calc_height(season, 1986, 2015, y)
  #x = np.arange(0,40,1)
  x = np.arange(0,50,0.5)
  X, Y= np.meshgrid(x, y)
  vmin=0
  vmax=15
  v = np.arange(vmin, vmax+1, 2)  

  fig, axes = plt.subplots(nrows=2, ncols=2, figsize=[28,16], sharex=True, sharey=True)

  for k, letter in zip(range(0,4), ['a', 'b', 'c', 'd']):
    subplt = '22{0}'.format(k+1)
    plt.subplot(subplt)

    CS = plt.contourf(X, Y, histo[k], v, cmap='cmo.haline', extend='max')
    CS.set_clim(vmin, vmax)
    plt.gca().invert_yaxis()
    plt.plot(centroids[k], y, color='white', marker='o', lw=4, markersize=10, markeredgecolor='k')
    #if (k%2) == 1:
          
    #CB = plt.colorbar(CS, extend='both', ticks=v)
    #CB.ax.tick_params(labelsize=20)
    plt.xlim(0,39)
    plt.ylim(min(y),max(y))
    plt.xticks(np.arange(0,50,5), fontsize=20)
    plt.yticks(np.arange(0,2401,200), fontsize=20)
    plt.title('({0}) {1:2.2f} % {2}'.format(letter, perc[k], shf[k]), fontsize='20')
  
  
  #cax,kw = mpl.colorbar.make_axes([ax for ax in axes.flat])
  cax = fig.add_axes([0.92, 0.1, 0.02, 0.8]) 
  CB = plt.colorbar(CS, cax=cax, extend='both', ticks=v)  
  CB.ax.tick_params(labelsize=20)
  #plt.tight_layout()
  plt.savefig('Images/{0}_{1}{2}_{3}.png'.format(name, datai, dataf, period), bbox_inches='tight')
  plt.close()
  #sys.exit()

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

  # Getting the probability distribution. Kernel Density  
  hist_0 = calc_kerneldensity(df_0)
  hist_1 = calc_kerneldensity(df_1)

  #print(np.mean(df_0, axis=0), np.mean(df_1, axis=0), kmeans.cluster_centers_)

  return kmeans.cluster_centers_, [hist_0, hist_1], [df_0.shape[0]*100/df_a.shape[0], df_1.shape[0]*100/df_a.shape[0]]

def calc_kerneldensity(df):
  hist_aux = []
  for i in range(0,8):
      kde_skl = KernelDensity(bandwidth=0.4)
      #aux = np.array(df_n['1000.0'])
      aux = np.copy(df[:,i])
      aux_grid2 = np.linspace(0,50,100)
      kde_skl.fit(aux[:, np.newaxis])
      log_pdf = kde_skl.score_samples(aux_grid2[:, np.newaxis])
      hist_aux.append(np.exp(log_pdf)*100)

  return hist_aux

def calc_histogram(df):

  hist_l = []
  bins = np.arange(0,50.25,1)
  for i in range(0, df.shape[1]):    
    hist, bins = np.histogram(df[:,i], bins=bins)
    hist_l.append(hist*100/sum(hist))

  return np.array(hist_l)
    




if __name__ == "__main__":
  main()
