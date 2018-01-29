#!/usr/bin/env python
# -*- coding: utf-8 -*-

## Time-averaged topographic representation of responses of cells in the
## network grids

import numpy as np
import matplotlib.pyplot as plt
import sys,os,os.path
import scipy.fftpack

# Data path
data_path = "/home/pablo/Desktop/Biophysical_thalamocortical_system/thalamocortical/results/"

# Number of neurons (all layers except INs)
N = 10.0

# Stimulus
stimulus = [4.0]

type = "patch_grating/kpg_015"
#type = "disk"

fig_num = "figure_1"
#IDs = ["RC-ON","RC-OFF","IN-ON","PY_h-ON","PY_v-ON"]
ID = "PY_v-ON"
#ID = "retinaON"

# Simulation parameters
tsim = 2000.0
binsize = 5.0
numbertrials =400.0

# Combination
cc = "comb0"
# model_type = 0 (phase-reversed FB), model_type = 1 (phase-matched FB)
model_type = 0

# Cells to plot
cell_number = 55

if os.path.isdir("tmp") == False:
    os.system("mkdir tmp")

# Load PST
def loadPST(stim,N,tsim,binsize,neuron,add_path):

    PST_avg = np.zeros((int(N*N),int(tsim/binsize)))
    lines = [line.rstrip('\n') for line in open(data_path+add_path+"/stim"+str(stim)+"/PST"+neuron, "r")]
    for n in np.arange(len(lines)):
        h = lines[int(n)].split(',')
        for pos in np.arange(0,tsim/binsize):
            PST_avg[int(n),int(pos)] = float(h[int(pos)])

    return PST_avg

# Create arrays of all PSTs
def createPST(cellID,stimulus,N,tsim,binsize,comb):

    PST = []

    if cellID=="retinaON":
        for s in stimulus:
            PST.append(loadPST(s,N,tsim,binsize,"","retina/"+type+"/"+"ON"))
    elif cellID=="retinaOFF":
        for s in stimulus:
            PST.append(loadPST(s,N,tsim,binsize,"","retina/"+type+"/"+"OFF"))
    else:
        for s in stimulus:
            PST.append(loadPST(s,N,tsim,binsize,cellID,"model_"+str(model_type)+"/"+type+"/"+fig_num+"/"+comb))

    return PST[0]

# Plots

# PSTs
if (ID == "retinaON" or ID=="retinaOFF"):
    PST = createPST(ID,stimulus,N,tsim,binsize,"")
else:
    PST = createPST(ID,stimulus,N,tsim,binsize,cc)

## PST
#fig = plt.figure()
#fig.subplots_adjust(hspace=0.4)

#Vax = plt.subplot2grid((1,1), (0,0))

#Vax.step(np.arange(0.0,tsim,binsize),PST[cell_number,:]/numbertrials)
#Vax.set_xlabel('time (ms)')
#Vax.set_ylabel('firing rate (Hz)')

## Save data to file
#np.savetxt('tmp/'+'PST_'+type[0]+'_model_'+str(model_type)+\
#'_'+str(stimulus[0])+'_'+ID+'.out', PST[cell_number,:]/numbertrials, delimiter=',')

#np.savetxt('tmp/'+'PST_xdata.out',np.arange(0.0,tsim,binsize), delimiter=',')

## Topographical responses

fig = plt.figure()
Gax = plt.subplot2grid((1,1), (0,0))

start = 750.0
stop = 1000.0
intervals = [int(start/binsize),int(stop/binsize)]
data = np.zeros((N,N))
n=0
for x in np.arange(N):
    for y in np.arange(N):
        data[int(x),int(y)] = np.sum(PST[int(n),intervals[0]:intervals[1]])/\
        (numbertrials*(intervals[1]-intervals[0]))
        n+=1

im1=Gax.matshow(data,vmin=0.0, vmax=40.0)
plt.setp(Gax, yticks=[])
plt.setp(Gax, xticks=[])
#yticklabels=['0', str(np.max(PST[Ncell,:]/self.numbertrials))])
#cbar1 = fig.colorbar(im1)

plt.show()


