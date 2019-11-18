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
  - Read the files with the projections for WED
'''

def main():

  lats = []
  lons = []
  stnames = []

  #parser = argparse.ArgumentParser()
  #parser.add_argument("datai", help="Initial year of the calculations", type=int)
  #parser.add_argument("dataf", help="Final year of the calculations", type=int)
  #args = parser.parse_args()  

  current = 'DatFiles/wind_density_data-1976-2005_v2.dat'

  projection = [(2000,2029),(2010,2039),(2020,2049),(2030,2059),(2040,2069),(2050,2079),(2060,2089),(2070,2099)]
  
     

  stations = open('DatFiles/stations.txt', 'r')
  for line in stations:
    aa = line.replace("\n", '').split(';')
    if (aa[0] != "#"):      
      lats.append(float(aa[3]))
      lons.append(float(aa[5]))
      stnames.append(aa[1].replace(',',"_"))  

  for name in stnames:    
        
    print(name)    
    df = pd.read_csv(current, skipinitialspace=True) 
    # Filtering the dataframe
    perc_t1, perc_t2, perc_t3, perc_t4, vmean_shfnt1, vmean_shfnt2, vmean_shfpt1, vmean_shfpt2 = dataframe_filter(df, name)
        
    perc_pj = []
    vmean_pj = []

    xaxis_l = []

    for di, df in projection:

      # read file
      f = 'DatFiles/wind_density_data-{0}-{1}_v2.dat'.format(di, df)

      df_pj = pd.read_csv(f, skipinitialspace=True)

      perc_t1_pj, perc_t2_pj, perc_t3_pj, perc_t4_pj, vmean_shfnt1_pj, vmean_shfnt2_pj, vmean_shfpt1_pj, vmean_shfpt2_pj = dataframe_filter(df_pj, name)

      perc_pj.append([perc_t1_pj, perc_t2_pj, perc_t3_pj, perc_t4_pj])
      vmean_pj.append([vmean_shfnt1_pj, vmean_shfnt2_pj, vmean_shfpt1_pj, vmean_shfpt2_pj])

      xaxis_l.append('{0}:{1}'.format(di, df))

    # Plot projections
    for season, sname in zip([[12, 1, 2],[6, 7, 8]], ['DJF','JJA']):
            
      # calculating season means for current      
      perc_t1_mean, perc_t2_mean, perc_t3_mean, perc_t4_mean = calc_smean(season, perc_t1, perc_t2, perc_t3, perc_t4)
      vmean_t1_mean, vmean_t2_mean, vmean_t3_mean, vmean_t4_mean = calc_smean(season, vmean_shfnt1, vmean_shfnt2, vmean_shfpt1, vmean_shfpt2)
      
      perc_mean_pj = []
      vmean_mean_pj = []
      #print(perc_t1_mean, perc_t2_mean, perc_t3_mean, perc_t4_mean)

      # calculating season means for future climate
      for i in range(0,len(perc_pj)):
        perc_aux1, perc_aux2, perc_aux3, perc_aux4 = calc_smean(season, perc_pj[i][0], perc_pj[i][1], perc_pj[i][2], perc_pj[i][3])
        vmean_aux1, vmean_aux2, vmean_aux3, vmean_aux4 = calc_smean(season, vmean_pj[i][0], vmean_pj[i][1], vmean_pj[i][2], vmean_pj[i][3])

        perc_mean_pj.append(np.array([perc_aux1, perc_aux2, perc_aux3, perc_aux4]))
        vmean_mean_pj.append(np.array([vmean_aux1, vmean_aux2, vmean_aux3, vmean_aux4]))
        
        #perc_mean_pj.append(perc_aux)
        #vmean_mean_pj.append(vmean_aux)

      #print(np.array(perc_mean_pj).shape)
      #print(np.array(perc_mean_pj))
      #sys.exit()
      
      #plot_projections([perc_t1_mean, perc_t2_mean, perc_t3_mean, perc_t4_mean], np.array(perc_mean_pj), sname, 'perc', xaxis_l, name)
      #plot_projections([vmean_t1_mean, vmean_t2_mean, vmean_t3_mean, vmean_t4_mean], np.array(vmean_mean_pj), sname, 'vmean', xaxis_l, name, True)

      plot_current_values([perc_t1_mean, perc_t2_mean, perc_t3_mean, perc_t4_mean], np.array(perc_mean_pj), sname, 'perc', xaxis_l, name)
      plot_current_values([vmean_t1_mean, vmean_t2_mean, vmean_t3_mean, vmean_t4_mean], np.array(vmean_mean_pj), sname, 'vmean', xaxis_l, name, True)

    #plot_monthly_weighted()
'''
    # Plot of the monthly Wind Energy Density - Average Weight
    months = range(1,13)

    plt.plot(months, vmean_shfnt1, label='SHF- t1')
    plt.plot(months, vmean_shfnt2, label='SHF- t2')
    plt.plot(months, vmean_shfpt1, label='SHF+ t1')
    plt.plot(months, vmean_shfpt2, label='SHF+ t2')
    plt.legend()
    plt.savefig('aaaaa.png')
    plt.close()

    print(perc_t1, perc_t2, perc_t3, perc_t4)

    # Plot of the monthly percentage of each type of PBL regime
    plt.plot(months, perc_t1, label='SHF- t1')
    plt.plot(months, perc_t2, label='SHF- t2')
    plt.plot(months, perc_t3, label='SHF+ t1')
    plt.plot(months, perc_t4, label='SHF+ t2')
    plt.legend()
    plt.savefig('bbbbb.png')
    plt.close()

    
    
    sys.exit()
    '''

def calc_smean(season, perc_t1, perc_t2, perc_t3, perc_t4):
  perc_t1_season = []
  perc_t2_season = []
  perc_t3_season = []
  perc_t4_season = []  

  for i in season:
    perc_t1_season.append(perc_t1[i-1])
    perc_t2_season.append(perc_t2[i-1])
    perc_t3_season.append(perc_t3[i-1])
    perc_t4_season.append(perc_t4[i-1])      

  return np.mean(perc_t1_season), np.mean(perc_t2_season), np.mean(perc_t3_season), np.mean(perc_t4_season)

def dataframe_filter(df, name):

  t1 = 'SHF-'
  t2 = 'SHF+'

  # Filtering the dataframe
  df_ = df.loc[(df['station'] == name)]      

  # Plotting for SHF- and Sorting it, to separate between the two PBL regimes          
  df_n = df_.loc[df_['type'] == t1].sort_values(by=['month', 'vmean'])
  df_p = df_.loc[df_['type'] == t2].sort_values(by=['month', 'vmean'])    

  perc_shfn_t1 = np.array(df_n['perc_kmean'][::2])
  perc_shfn_t2 = np.array(df_n['perc_kmean'][1::2])

  perc_shfp_t1 = np.array(df_p['perc_kmean'][::2])
  perc_shfp_t2 = np.array(df_p['perc_kmean'][1::2])

  perc_shfn = np.array(df_n['perc_shf'][::2])
  perc_shfp = np.array(df_p['perc_shf'][::2])   

  # Percentage and Weighted Vmean for current climate (1976-2005)
  # t1: SHF-, t1; t2: SHF-, t2; t3: SHF+, t1; t4: SHF+, t2
  perc_t1 = perc_shfn_t1*perc_shfn/100
  perc_t2 = perc_shfn_t2*perc_shfn/100

  perc_t3 = perc_shfp_t1*perc_shfp/100
  perc_t4 = perc_shfp_t2*perc_shfp/100

  vmean_shfnt1 = np.array(df_n['vmean'][::2]) * perc_t1 /100
  vmean_shfnt2 = np.array(df_n['vmean'][1::2]) * perc_t2 / 100

  vmean_shfpt1 = np.array(df_p['vmean'][::2]) * perc_t3 / 100
  vmean_shfpt2 = np.array(df_p['vmean'][1::2]) * perc_t4 / 100  

  return perc_t1, perc_t2, perc_t3, perc_t4, vmean_shfnt1, vmean_shfnt2, vmean_shfpt1, vmean_shfpt2

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

def plot_projections(vcurrent, v_proj, sname, name, x, lc, ty=False):

  fig = plt.figure(figsize=[14,8])
#  print(vcurrent, v_proj)

  ll = ['VSBL', 'WSBL', 'UNST', 'SHEAR']

  plt.plot(x, [0,0,0,0,0,0,0,0], color='k')
  for i in range(0, len(vcurrent)):

    plt.plot(x, v_proj[:,i] - vcurrent[i], label=ll[i], marker='o')
  
  #plt.plot(centroids[k], y, color='white', marker='o', lw=4, markersize=10, markeredgecolor='k')
  
  #plt.xlim(0,800)
  #plt.ylim(1000,700)
  plt.xticks(fontsize=20, rotation=45)
  plt.yticks(fontsize=20)
  if ty:
    plt.ylabel('Change in Weighted Average WED (W/m2)', fontsize=20)
  else:
    plt.ylabel('Change in Frequency (%)', fontsize=20)
  #plt.title('({0}) {1:2.2f} % {2}'.format(letter, perc[k], shf[k]), fontsize='20')
  plt.legend()
  #plt.tight_layout()
  plt.savefig('Images/Proj/{2}_proj_{0}_{1}.png'.format(name, sname, lc), pad_inches=0.0, bbox_inches='tight')
  plt.close()
  #sys.exit()

  return None  

def plot_current_values(vcurrent, v_proj, sname, name, x, lc, ty=False):

  fig = plt.figure(figsize=[8,8])
#  print(vcurrent, v_proj)

  ll = ['VSBL', 'WSBL', 'UNST', 'SHEAR']

  plt.bar(ll, vcurrent)
  
  #plt.plot(centroids[k], y, color='white', marker='o', lw=4, markersize=10, markeredgecolor='k')
  
  #plt.xlim(0,800)
  #plt.ylim(1000,700)
  plt.xticks(fontsize=20, rotation=45)
  plt.yticks(fontsize=20)
  if ty:
    plt.ylabel('Weighted Average WED (W/m2)', fontsize=20)
  else:
    plt.ylabel('Frequency (%)', fontsize=20)
  #plt.title('({0}) {1:2.2f} % {2}'.format(letter, perc[k], shf[k]), fontsize='20')
  plt.legend()
  #plt.tight_layout()
  plt.savefig('Images/Proj/{2}_proj_{0}_{1}_Current.png'.format(name, sname, lc), pad_inches=0.0, bbox_inches='tight')
  plt.close()
  #sys.exit()

  return None    

if __name__ == "__main__":
  main()
