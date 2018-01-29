#!/usr/bin/env python
# -*- coding: utf-8 -*-

## x-y-t receptive field maps

import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.ndimage.filters import gaussian_filter

# data path
data_path = "/home/pablo/Desktop/Biophysical_thalamocortical_system/thalamocortical/results/"

##################
### Parameters ###
##################

# Number of neurons (all layers except INs)
N = 10.0

# Stimulus
# Spatial impulse response
Npoints = 40.0 # Number of flashing spots per row
raw_stimulus = np.arange(0.0,Npoints*Npoints,1.0)
stimulus = []
# To speed computations, select a center square
row_top_limit = 3.0 # (between 0 and N)
row_bottom_limit = 7.0 # (between 0 and N)
col_left_limit = 3.0 # (between 0 and N)
col_right_limit = 7.0 # (between 0 and N)

for pos in raw_stimulus:
    row = int(pos/Npoints)*(N/Npoints)
    col = np.remainder(pos,Npoints)*(N/Npoints)

    if(row >= row_top_limit and row <= row_bottom_limit and col >=\
    col_left_limit and col <= col_right_limit):
        stimulus.append(pos)

#ID = "retinaON"
#ID = "RC-ON"
ID = "PY_v-ON"

# Simulation parameters
tsim = 300.0
binsize = 5.0
numbertrials =100.0

# Combination
cc = "comb0"
selected_cell = 55

##################
### Plots ########
##################

# Load PST
def loadPST(stim,N,tsim,binsize,neuron,add_path):

    PST_avg = np.zeros((int(N*N),int(tsim/binsize)))
    lines = [line.rstrip('\n') for line in open(data_path+add_path+"/stim"+str(stim)+"/PST"+neuron, "r")]
    for n in np.arange(len(lines)):
        h = lines[int(n)].split(',')
        for pos in np.arange(0,tsim/binsize):
            PST_avg[int(n),int(pos)] = float(h[int(pos)])

    return PST_avg

# Create PSTs
def createPST(cellID,stimulus,N,tsim,binsize,comb,type):

    PST = []

    if cellID=="retinaON":
        for s in stimulus:
            PST.append(loadPST(s,N,tsim,binsize,"","retina/"+type+"/"+"ON"))
    elif cellID=="retinaOFF":
        for s in stimulus:
            PST.append(loadPST(s,N,tsim,binsize,"","retina/"+type+"/"+"OFF"))
    else:
        for s in stimulus:
            PST.append(loadPST(s,N,tsim,binsize,cellID,"xt_plots/"+type+"/"+comb))

    return PST[0]

# xt map
def xt_map(intervals,type):
    data = np.zeros((np.sqrt(len(stimulus)),np.sqrt(len(stimulus))))

    n = 0
    for x in np.arange(0,np.sqrt(len(stimulus))):
        for y in np.arange(0,np.sqrt(len(stimulus))):
            # PST
            if (ID == "retinaON" or ID=="retinaOFF"):
                PST = createPST(ID,[stimulus[n]],N,tsim,binsize,"",type)
            else:
                PST = createPST(ID,[stimulus[n]],N,tsim,binsize,cc,type)

            data[int(x),int(y)] = np.sum(PST[int(selected_cell),intervals[0]:intervals[1]])/\
            (numbertrials*(intervals[1]-intervals[0]))

            n+=1

    return data

# Spatiotemporal RF
def spatiotemporalRF(intervals,vertical):

    data = np.zeros((len(intervals)-1,np.sqrt(len(stimulus))))

    for n in np.arange(0,len(intervals)-1):
        print("time = %s" % intervals[n])
        data_1 = xt_map([int(intervals[n]/binsize),int(intervals[n+1]/binsize)],"RF_1_matched_old")
        data_2 = xt_map([int(intervals[n]/binsize),int(intervals[n+1]/binsize)],"RF_2_matched_old")
        aux_data = data_1 - data_2

        for y in np.arange(0,np.sqrt(len(stimulus))):
            if(vertical):
                int_data = np.sum(aux_data[y,:])/np.sqrt(len(stimulus))
            else:
                int_data = np.sum(aux_data[:,y])/np.sqrt(len(stimulus))

            data[n,y] = int_data

    return data


### Plotting ###

## Topographical responses
start = 200.0
stop = 220.0
intervals = [int(start/binsize),int(stop/binsize)]
print("computing ON response")
data_1 = xt_map(intervals,"RF_1_matched_old")
print("computing OFF response")
data_2 = xt_map(intervals,"RF_2_matched_old")

data = data_1 - data_2

x = np.arange(-2.0, 2.25, 0.25)
y = np.arange(-2.0, 2.25, 0.25)
#x = np.arange(Npoints)
#y = np.arange(Npoints)
X, Y = np.meshgrid(x, y)

### Spatiotemporal response
#start = 100.0
#stop = 300.0
#intervals = np.arange(start,stop,10.0)

#data = spatiotemporalRF(intervals,False)

#x = np.arange(-2.0, 2.25, 0.25)
#y = intervals[0:len(intervals)-1]
#X, Y = np.meshgrid(x, y)

#### Contour plot
fig = plt.figure()
Gax = plt.subplot2grid((1,1), (0,0))

# interpolation
sigma = 1.5
data2 = gaussian_filter(data, sigma)

#CS = plt.contourf(X, Y, data2, 11, cmap=plt.cm.coolwarm)
CS = plt.contourf(X, Y, data2, [-13.5,-13.,-11.3,-7.5,-6.,-4.5,-3.,-1.5,0.,1.5,3.,4.5], cmap=plt.cm.coolwarm)
plt.setp(Gax, yticks=[])
plt.setp(Gax, xticks=[])
cbar = plt.colorbar(CS,ticks=[])
#cbar = plt.colorbar(CS)
#plt.rcParams.update({'font.size': 22})

#### x/y profiles
#fig = plt.figure()
#plt.plot(np.arange(-2.0, 2.25, 0.25),data[8,:])
#plt.plot(np.arange(-2.0, 2.25, 0.25),data[:,8])

plt.show()


