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
  #df_2030 = pd.read_csv('wind_density_data-2020-2049.dat', skipinitialspace=True)
  #df_2050 = pd.read_csv('wind_density_data-2040-2069.dat', skipinitialspace=True)
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
      df_2080_ = df_2080.loc[(df_2080['station'] == name)]      
      
      # Plotting for SHF- and Sorting it, to separate between the two PBL regimes
      df2_sorted = df_.loc[df_['type'] == shf_type].sort_values(by=['month', 'vmean'])      
      df_2080_sorted = df_2080_.loc[df_2080_['type'] == shf_type].sort_values(by=['month', 'vmean']) 
      
      perc_vsbl = df2_sorted['perc_kmean'][::2]
      perc_wsbl = df2_sorted['perc_kmean'][1::2]
      
      perc_vsbl_2080 = df_2080_sorted['perc_kmean'][::2]
      perc_wsbl_2080 = df_2080_sorted['perc_kmean'][1::2]
      
      value_vsbl = df2_sorted['vmean'][::2]
      value_wsbl = df2_sorted['vmean'][1::2]

      std_vsbl = df2_sorted['vstd'][::2]
      std_wsbl = df2_sorted['vstd'][1::2]

      value_vsbl_future = df_2080_sorted['vmean'][::2]
      value_wsbl_future = df_2080_sorted['vmean'][1::2]

      std_vsbl_future = df_2080_sorted['vstd'][::2]
      std_wsbl_future = df_2080_sorted['vstd'][1::2]
          
      # Barplot with the 2 percentages
      
      # On the right side, climate projections
      barplot_perc(np.arange(1,13), perc_vsbl.values, perc_wsbl.values, perc_vsbl_2080.values, perc_wsbl_2080.values,
            'VSBL: Current Climate', 'WSBL: Current Climate', 
            'VSBL: {0}:{1} - Current'.format(datai, dataf), 'WSBL: {0}:{1} - Current'.format(datai, dataf),
            '{0} Percentage of occurance of PBL Regimes, {1}, RCP8.5'.format(name[7:-4], shf_type),
            'Images_Bar/{0}_perc_{2}_{1}.png'.format(name, datai, shf_type))

      barplot_perc_all(np.arange(1,13), perc_vsbl.values, perc_wsbl.values, perc_vsbl_2080.values, perc_wsbl_2080.values,
                  'VSBL: Current Climate', 'WSBL: Current Climate', 
                  'VSBL: {0}:{1}'.format(datai, dataf), 'WSBL: {0}:{1}'.format(datai, dataf),
                  'VSBL: {0}:{1} - Current'.format(datai, dataf), 'WSBL: {0}:{1} - Current'.format(datai, dataf),
                  '{0} Percentage of occurance of PBL Regimes, {1}, RCP8.5'.format(name[7:-4], shf_type),
                  'Images_Bar/{0}_perc_{2}_{1}_all.png'.format(name, datai, shf_type))
      
    # Plot 2
    # Lineplot with the 2 values
      plot_wind_power(np.arange(1,13), value_vsbl.values, std_vsbl.values, value_wsbl.values, std_wsbl.values,
            value_vsbl_future.values, std_vsbl_future.values, value_wsbl_future.values, std_wsbl_future.values,
            'VSBL: Current Climate', 'WSBL: Current Climate', 
            'VSBL: {0}:{1}'.format(datai, dataf), 'WSBL: {0}:{1}'.format(datai, dataf),
            'VSBL: {0}:{1} - Current'.format(datai, dataf), 'WSBL: {0}:{1} - Current'.format(datai, dataf),
            '{0} Wind Power Density (W/m2), {1}, RCP8.5'.format(name[7:-4], shf_type),
            'Images_Bar/{0}_WPD_{2}_{1}_all.png'.format(name, datai, shf_type))
    # On the right side, climate projections
      #sys.exit() 
    


def plot_wind_power(xdata, ydata1, yerr1, ydata2, yerr2, 
                 ydata1_future, yerr1_future, ydata2_future, yerr2_future,
                 label1, label2, label1_future, label2_future,
                 label1_proj, label2_proj, title, fname, barWidth=0.4):  
    
  fig = plt.figure(figsize=[14,8])
  x1 = xdata
  x2 = xdata + barWidth/1.5
  #print(x2)
    
  ax1 = fig.add_subplot(111)
    
  plt.plot([0,13],[0,0], color='k', linewidth=1)
    
  # Create blue bars
  plt.bar(x1 - barWidth - barWidth/4, ydata1, yerr=yerr1, width = barWidth/2, color = 'royalblue', edgecolor = 'black', label=label1)
  plt.bar(x1 - barWidth/2 - barWidth/4, ydata1_future, yerr=yerr1_future, width = barWidth/2, color = 'darkblue', edgecolor = 'black', label=label1_future)

    # Create cyan bars
  plt.bar(x1 - barWidth/4, ydata2, yerr=yerr2, width = barWidth/2, color = 'firebrick', edgecolor = 'black', label=label2)
  plt.bar(x1 + barWidth/2 - barWidth/4, ydata2_future, yerr=yerr2_future, width = barWidth/2, color = 'darkred', edgecolor = 'black', label=label2_future)
    
  plt.ylim(-1500,7000)        
  plt.xlim(0,13)
  plt.xticks(xdata - barWidth/2, ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], fontsize=20)    
  plt.xlabel('Months', fontsize='20')
    
  ax1.set_yticks([], [])
  plt.legend(loc=1, fontsize=16)
    
  ax2 = fig.add_subplot(111, sharex=ax1, frameon=False, )
  plt.bar(x1 - barWidth, ydata1_future-ydata1, width = barWidth, color = 'olivedrab', edgecolor = 'black', label=label1_proj)
    
  plt.bar(x1, ydata2_future-ydata2, width = barWidth, color = 'goldenrod', edgecolor = 'black', label=label2_proj)
  plt.ylim(-1500,7000)
  plt.xlim(0,13)
  plt.setp(ax2.get_xticklabels(), visible=False)
  plt.yticks(np.arange(-1500,7001,500), fontsize=20)
    
  plt.title(title, fontsize=20)
    
  plt.ylabel('Wind Power Density (W/m2)', fontsize=20)
  plt.legend(loc=2, fontsize=16)
  plt.savefig(fname, pad_inches=0.0, bbox_inches='tight')   
  plt.close()


def barplot_perc(xdata, ydata1, ydata2, ydata1_future, ydata2_future, 
                 label1, label2, label1_future, label2_future, title, fname, barWidth=0.5):  
    
  fig = plt.figure(figsize=[14,8])
  x1 = xdata
  x2 = xdata + barWidth/1.5
  #print(x2)
    
  ax1 = fig.add_subplot(111)
    
  plt.plot([0,13],[0,0], color='k', linewidth=1)
    
  # Create blue bars
  plt.bar(x1, ydata1, width = barWidth/1.5, color = 'royalblue', edgecolor = 'black', label=label1)
 
    # Create cyan bars
  plt.bar(x2, ydata2, width = barWidth/1.5, color = 'firebrick', edgecolor = 'black', label=label2)
    
  plt.ylim(-15,100)        
  plt.xlim(0,13)
  plt.xticks(xdata+barWidth/3, ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], fontsize=20)    
  plt.xlabel('Months', fontsize='20')
    
  ax1.set_yticks([], [])
  plt.legend(loc=1, fontsize=16)
    
  ax2 = fig.add_subplot(111, sharex=ax1, frameon=False, )
  plt.bar(x1, ydata1_future-ydata1, width = barWidth/1.5, color = 'olivedrab', edgecolor = 'black', label=label1_future)
    
  plt.bar(x2, ydata2_future-ydata2, width = barWidth/1.5, color = 'goldenrod', edgecolor = 'black', label=label2_future)
  plt.ylim(-15,100)
  plt.xlim(0,13)
  plt.setp(ax2.get_xticklabels(), visible=False)
  plt.yticks(np.arange(-10,101,10), fontsize=20)
    
  plt.title(title, fontsize=20)
    
  plt.ylabel('Frequency (%)', fontsize=20)
  plt.legend(loc=2, fontsize=16)
  plt.savefig(fname, pad_inches=0.0, bbox_inches='tight')   
  plt.close()

def barplot_perc_all(xdata, ydata1, ydata2, ydata1_future, ydata2_future, 
                 label1, label2, label1_future, label2_future, 
                 label1_proj, label2_proj, title, fname, barWidth=0.4):  
    
  fig = plt.figure(figsize=[14,8])
  x1 = xdata
  x2 = xdata + barWidth/1.5
  #print(x2)
    
  ax1 = fig.add_subplot(111)
    
  plt.plot([0,13],[0,0], color='k', linewidth=1)
    
  # Create blue bars
  plt.bar(x1 - barWidth - barWidth/4, ydata1, width = barWidth/2, color = 'royalblue', edgecolor = 'black', label=label1)
  plt.bar(x1 - barWidth/2 - barWidth/4, ydata1_future, width = barWidth/2, color = 'darkblue', edgecolor = 'black', label=label1_future)

    # Create cyan bars
  plt.bar(x1 - barWidth/4, ydata2, width = barWidth/2, color = 'firebrick', edgecolor = 'black', label=label2)
  plt.bar(x1 + barWidth/2 - barWidth/4, ydata2_future, width = barWidth/2, color = 'darkred', edgecolor = 'black', label=label2_future)
    
  plt.ylim(-15,100)        
  plt.xlim(0,13)
  plt.xticks(xdata - barWidth/2, ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], fontsize=20)    
  plt.xlabel('Months', fontsize='20')
    
  ax1.set_yticks([], [])
  plt.legend(loc=1, fontsize=16)
    
  ax2 = fig.add_subplot(111, sharex=ax1, frameon=False, )
  plt.bar(x1 - barWidth, ydata1_future-ydata1, width = barWidth, color = 'olivedrab', edgecolor = 'black', label=label1_proj)
    
  plt.bar(x1, ydata2_future-ydata2, width = barWidth, color = 'goldenrod', edgecolor = 'black', label=label2_proj)
  plt.ylim(-15,100)
  plt.xlim(0,13)
  plt.setp(ax2.get_xticklabels(), visible=False)
  plt.yticks(np.arange(-10,101,10), fontsize=20)
    
  plt.title(title, fontsize=20)
    
  plt.ylabel('Frequency (%)', fontsize=20)
  plt.legend(loc=2, fontsize=16)
  plt.savefig(fname, pad_inches=0.0, bbox_inches='tight')   
  plt.close()
    
if __name__ == "__main__":
  main()



