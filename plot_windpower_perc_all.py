#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def main():
# Read the file from present and future
  datai = 2070
  dataf = 2099

  df = pd.read_csv('wind_density_data-1976-2005.dat', skipinitialspace=True)
  df_2030 = pd.read_csv('wind_density_data-2020-2049.dat', skipinitialspace=True)
  df_2050 = pd.read_csv('wind_density_data-2040-2069.dat', skipinitialspace=True)
  df_2080 = pd.read_csv('wind_density_data-{0}-{1}.dat'.format(datai, dataf), skipinitialspace=True)


# Get the name of the stations
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


# Plot for each station
  for shf_type in ['SHF-', 'SHF+']:
    for name in stnames:
      name_ = name
      
      # Filtering the dataframe
      df_ = df.loc[(df['station'] == name)]
      df_2030_ = df_2030.loc[(df_2080['station'] == name)]
      df_2050_ = df_2050.loc[(df_2080['station'] == name)]
      df_2080_ = df_2080.loc[(df_2080['station'] == name)]      
      
      # Plotting for SHF- and Sorting it, to separate between the two PBL regimes    
      
      df2_sorted = df_.loc[df_['type'] == shf_type].sort_values(by=['month', 'vmean'])
      df_2030_sorted = df_2030_.loc[df_2050_['type'] == shf_type].sort_values(by=['month', 'vmean'])  
      df_2050_sorted = df_2050_.loc[df_2030_['type'] == shf_type].sort_values(by=['month', 'vmean'])  
      df_2080_sorted = df_2080_.loc[df_2080_['type'] == shf_type].sort_values(by=['month', 'vmean'])  
      
      perc_base = df2_sorted['perc_shf'][::2]
      perc_2030 = df_2030_sorted['perc_shf'][::2]
      perc_2050 = df_2050_sorted['perc_shf'][::2]
      perc_2080 = df_2080_sorted['perc_shf'][::2]            
          
      # Barplot with the 4 percentages
      
      # On the right side, climate projections
      barplot_perc(np.arange(1,13), perc_base.values, perc_2030.values, perc_2050.values, perc_2080.values,
                  '1976:2005', '2020:2049', '2040:2069', '2070:2099',
                  '{0} Percentage of {1}, RCP8.5'.format(name[7:-4], shf_type),
                  'Images_Bar/{0}_perc_all_{1}.png'.format(name, shf_type))
      
      #sys.exit() 
      # Plot 2
      # Lineplot with the 2 values
      
      # On the right side, climate projections
    


#def lineplot_wp(xdata, ydata1, ydata2, ydata1_future, ydata2_future, 
#                 label1, label2, label1_future, label2_future, title, fname, barWidth=0.5):


# In[149]:


def barplot_perc(xdata, ydata1, ydata2, ydata3, ydata4, 
                 label1, label2, label3, label4, title, fname, barWidth=0.2):  
    
  fig = plt.figure(figsize=[14,8])
  x1 = xdata    
    
  ax1 = fig.add_subplot(111)
    
  plt.plot([0,13],[0,0], color='k', linewidth=1)
    
  # Create blue bars
  plt.bar(x1 - 2*barWidth, ydata1, width = barWidth, color = '#e41a1c', edgecolor = 'black', label=label1)   
  plt.bar(x1 - barWidth, ydata2, width = barWidth, color = '#377eb8', edgecolor = 'black', label=label2)
  plt.bar(x1, ydata3, width = barWidth, color = '#4daf4a', edgecolor = 'black', label=label3)
  plt.bar(x1 + barWidth, ydata4, width = barWidth, color = '#ff7f00', edgecolor = 'black', label=label4)
    
  plt.ylim(-15,100)        
  plt.xlim(0,13)
  plt.xticks(xdata - barWidth/2, ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], fontsize=20)    
  plt.xlabel('Months', fontsize='20')
    
  #ax1.set_yticks([], [])
  #plt.legend(loc=1, fontsize=16)
    
  #ax2 = fig.add_subplot(111, sharex=ax1, frameon=False, )
  #plt.bar(x1, ydata1_future-ydata1, width = barWidth/1.5, color = 'olivedrab', edgecolor = 'black', label=label1_future)
    
  #plt.bar(x2, ydata2_future-ydata2, width = barWidth/1.5, color = 'goldenrod', edgecolor = 'black', label=label2_future)
  plt.ylim(-15,100)
  plt.xlim(0,13)
  #plt.setp(ax2.get_xticklabels(), visible=False)
  plt.yticks(np.arange(-10,101,10), fontsize=20)
    
  plt.title(title, fontsize=20)
    
  plt.ylabel('Frequency (%)', fontsize=20)
  plt.legend(loc=2, fontsize=16)
  plt.savefig(fname, pad_inches=0.0, bbox_inches='tight')   
  fig.close()
  
    
if __name__ == "__main__":
  main()



