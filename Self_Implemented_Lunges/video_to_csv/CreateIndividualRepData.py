#!/usr/bin/env python
# coding: utf-8

# This code expects the csv files from individual videos to be present inside this directory structure:
#           video_to_csv
#                 |
#                 |
#         csv(combined_reps)
#           /           \
#          /             \
#        test           train
#       /  |  \        /  |  \
#      /   |   \      /   |   \
#     b1   b2   g1   b1   b2  g1
#  <------       csv     -------> (inside each directory)

# And arranges combines multiple csv files inside the same directory into a single Excel file
# whilst splitting them into individial sheets for each rep in this directory structure
#           video_to_csv
#                 |
#                 |
#           processed_data
#           /           \
#          /             \
#        test           train
#  <------  excel files  ------>


import os
import random
import time
import pandas as pd
import statsmodels.api as sm

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

from scipy.signal import find_peaks, peak_prominences, chirp, peak_widths

from os import walk
from pathlib import Path


USE_CUDA = True
if torch.cuda.is_available():  
  dev = "cuda:0" 
else:  
  dev = "cpu"  


import matplotlib.pyplot as plt


def find_before(val,index,data):
  '''
  This takes val as target value before which other value should be smaller,
  index as which before which it should look
  and data as list.
  '''
  ans = index
  for i in range(index,-1,-1):
    if data[i] > val:
      ans = i
    else:
      break
  return ans

def find_after(val,index,data):
  '''
  This takes val as target value from which other value should be smaller,
  index as which after which it should look
  and data as list.
  '''
  ans = index
  for i in range(index,len(data)):
    if data[i] > val:
      ans = i
    else:
      break
  return ans





def fpeak(data,plot = False):
  y = 1-np.array( (data.iloc[:,4+4*0+3]).tolist() )# left wrist index 15 y coord, subtract from 1 as pose gan give headtopmost position smaller value
#   t=np.arange(20)
#   y=np.sin(t)
#   print(y[:30])
 
  peaks,_ = find_peaks(y)
  prominences = peak_prominences(y, peaks)[0]
  std_dev = prominences.std()
    
  ## single end start.
  # good_peaks = [0]
  good_peaks = [[0]]

  for i in range(len(prominences)):
    if prominences[i]  > std_dev:
      # seprate end and start
      key = y[peaks[i]] - ( (prominences[i]) * 0.1 )
#       print(peaks[i])
#       print(prominences[i])
      e0 = find_before(key,peaks[i],y)
      good_peaks[-1].append(e0)

      s1 = find_after(key,peaks[i],y)
      good_peaks.append([s1])

      ## single end start.
      # good_peaks.append(peaks[i])

  # seprate end and start     
  good_peaks[-1].append(len(y))
  good_peaks = good_peaks[1:-1] # remove first and last peaks, prone to errors
  ## if single start stop
  # good_peaks.append(len(y))
  print("good peaks count:" , len(good_peaks))
  if plot:
    print("plotting")
    contour_heights = y[peaks] -  prominences
      
    fig,ax=plt.subplots(figsize=(20, 15))
    plt.plot(np.arange(0,len(y)),y,c='y')
    pt=[peaks[i]  for i in range(len(peaks))  if prominences[i]  > std_dev]
#     Good peakso only
    # plt.scatter(pt, y[pt], c= 'b')
    plt.scatter(peaks, y[peaks], c= 'b')

    #plt.scatter(start_end_x, start_end_y, c= 'y')
      
    plt.vlines(x=peaks, ymin=contour_heights, ymax= y[peaks], color = 'red')
      
      #plt.hlines(*results_half[1:],color = 'purple')
      #plt.hlines(*results_full[1:],color = 'black')
      
    plt.legend(['real','peaks','prominences','half width', 'full width'])
#     plt.close()
    plt.show()

  # single end start.
  #good_peaks.append(len(y))
  return good_peaks


print("Executing CreateInidividualRepData!")
for (root, dirs, filenames) in walk('./csv(combined_reps)'):
    for dir in dirs:
        for (label_dir, _, filenames) in walk(os.path.join(root, dir)):
            print(label_dir)
            if(filenames == []): #Since we are walking in directory './csv(combined_reps)' and will save an excel called 'test.xlsx' if we don't skip this step
                continue
            
            writer = pd.ExcelWriter(os.path.join('processed_data/'+ label_dir[21:-2]) + label_dir[-2:]+ '.xlsx', engine='xlsxwriter')
            sheet_count=0
            
            for filename in filenames:
                # f.append(os.path.join(dir, filename))
                filepath = os.path.join(label_dir, filename)
                print(filepath)

                df=pd.read_csv(filepath,header=None)
                peaks = fpeak( df,False)
                pre=[]
                file_index = 1
                for i in range(len(peaks)-1):
                    inpt = df.iloc[peaks[i],:]
                    pre_val=[]

                    temp =  df.iloc[peaks[i][0]:peaks[i][1]]
                    ## if single start stop
                    # temp =  df.iloc[peaks[i]:peaks[i+1]]

                    temp.to_excel(writer,sheet_name=str(sheet_count), index=False)
                    sheet_count+=1
            writer._save()
        # print(dir)
    break

print("Successfully split the reps from csv file into different sheets on Excel")
