#!/usr/bin/env python
# -*- coding: utf-8 -*-

## Area-response curves for different corticothalamic synapse weights and spatial
## connectivity profiles of cortical feedback

import numpy as np
import matplotlib.pyplot as plt
import sys,os,os.path
import scipy.fftpack
from scipy.ndimage.filters import gaussian_filter

# Data path
data_path = "/home/pablo/Desktop/Biophysical_thalamocortical_system/thalamocortical/results/"

# Number of neurons (all layers except INs)
N = 10.0

# Stimulus diameters
stimulus = np.arange(0.0,10.2,0.2)

# Folder
folder = "retina/disk/ON"

# Type of stimulus (disk/patch)
type = "disk"

IDs = ["RC-ON"]

# Simulation parameters
tsim = 1000.0
binsize = 5.0
numbertrials =100.0

# Interval to average disk response
spot_interval = [500.0,1000.0]

# Combinations
combinations = ["comb0","comb1","comb2","comb3",
"comb4","comb5","comb6","comb7",
"comb8","comb9","comb10","comb11",
"comb12","comb13","comb14","comb15"]

# Cells to plot
cell_number = 55

if os.path.isdir("tmp") == False:
    os.system("mkdir tmp")

# Metrics: center-surround antagonism
def alphavr(response,stimulus):

    rc = np.max(response)
    rc_pos = np.argmax(response)
    rcs = np.min(response[rc_pos:])
    rcs_pos = np.argmin(response[rc_pos:])+rc_pos
    alpha = 100.0 * (rc - rcs) / (rc - response[0])

    print("Stimulus[rc] = %s, rc = %s" % (stimulus[rc_pos],rc))
    print("Stimulus[rcs] = %s, rcs = %s" % (stimulus[rcs_pos],rcs))
    print("alpha_vr = %s" % alpha)

    return stimulus[rc_pos], alpha

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
def createPST(cellID,stimulus,N,tsim,binsize,add_path):

    PST = []
    for s in stimulus:
        PST.append(loadPST(s,N,tsim,binsize,cellID,add_path))

    return PST

# Area-response curve
def area_response(PSTs,cell_n):

    response = []
    # Estimation of spontaneous rate
    sp_rate = np.sum((PSTs[0])[cell_n,:])/(len((PSTs[0])[cell_n,:])*numbertrials)

    for PST in PSTs:
        if(type == "disk"):
            PST = PST[cell_n,int(spot_interval[0]/binsize):int(spot_interval[1]/binsize)]/numbertrials
            response.append(np.sum(PST)/len(PST))
        else:
            # DC response is calculated for each diameter as the mean response over
            # a time interval
            PST = PST[cell_n,int(250.0/binsize):int(1250.0/binsize)]/numbertrials
            response.append( np.mean(np.abs(PST - np.mean(PST))) + np.mean(PST))

    return response

# 7-point interpolation
def interpolation(response,stimulus):

    new_response = [response[i]+response[i+1]+response[i+2]+response[i+3]+\
    response[i+4]+response[i+5]+response[i+6] for i in np.arange(len(stimulus)-6)]
    new_response = np.array(new_response)/7.0
    xdata = stimulus[3:len(stimulus)-3]

    # For d = 0
    xdata = np.concatenate((np.array([0.0]),np.array(xdata)))
    new_response = np.concatenate((np.array([response[0]]),new_response))

    return xdata,new_response

# Plots
fig1 = plt.figure(1)
fig1.subplots_adjust(hspace=0.4)

fig2 = plt.figure(2)
fig2.subplots_adjust(hspace=0.4)

NoFB_response_abs = []
NoFB_response_norm = []

row = 0
col = 0

alpha_table = np.zeros((4,4))
dc_table = np.zeros((4,4))

for cc in combinations:

    print(cc)

    for ID in IDs:

        # PSTs
        PST = createPST(ID,stimulus,N,tsim,binsize,folder+str(cc))

        # Responses
        response = area_response(PST,cell_number)

        # Interpolated responses
        xdata,new_response = interpolation(response,stimulus)

        # Absolute response
        plt.figure(1)
        Vax = plt.subplot2grid((4,4), (row,col))
        Vax.plot(xdata,new_response,'k',label = ID)

        # Alpha coefficient
        rc_pos, alpha = alphavr(new_response,xdata)
        alpha_table[row,col] = alpha
        dc_table[row,col] = rc_pos

        if(row==0 and col==0):
            NoFB_response_abs = new_response

        # No-FB response
        Vax.plot(xdata,NoFB_response_abs,'r',label = "No FB")

        # Save data to file
        np.savetxt('tmp/'+'area_response_abs_'+type+\
        '_'+str(cc)+'_'+ID+'.out', new_response, delimiter=',')

        # Normalized response
        plt.figure(2)
        Gax = plt.subplot2grid((4,4), (row,col))
        new_response = new_response - np.min(new_response)
        Gax.plot(xdata,new_response/np.max(new_response),'k',label = ID)

        if(row==0 and col==0):
            NoFB_response_norm = new_response/np.max(new_response)

        # No-FB response
        Gax.plot(xdata,NoFB_response_norm,'r',label = "No FB")

        # Save data to file
        np.savetxt('tmp/'+'area_response_norm_'+type+\
        '_'+str(cc)+'_'+ID+'.out', new_response/np.max(new_response), delimiter=',')

        np.savetxt('tmp/'+'area_response_xdata.out',xdata, delimiter=',')

    # labels
    if(row==0 and col==0):
        Vax.set_xlabel('PY-IN: 0.0 nS')
        Vax.xaxis.set_label_position('top')
    if(row==0 and col==1):
        Vax.set_xlabel('PY-IN: 0.5 nS')
        Vax.xaxis.set_label_position('top')
    if(row==0 and col==2):
        Vax.set_xlabel('PY-IN: 1.0 nS')
        Vax.xaxis.set_label_position('top')
    if(row==0 and col==3):
        Vax.set_xlabel('PY-IN: 6.0 nS')
        Vax.xaxis.set_label_position('top')

    if(row==0 and col==0):
        h = Vax.set_ylabel('PY-RC: 0.0 nS')
#        h.set_rotation(0)
    if(row==1 and col==0):
        h = Vax.set_ylabel('PY-RC: 1.0 nS')
#        h.set_rotation(0)
    if(row==2 and col==0):
        h = Vax.set_ylabel('PY-RC: 2.0 nS')
#        h.set_rotation(0)
    if(row==3 and col==0):
        h = Vax.set_ylabel('PY-RC: 6.0 nS')
#        h.set_rotation(0)

    if(row==3):
        Vax.set_xlabel('Spot diameter (deg)')

    if (row==3 and col==3):
        Vax.legend(loc=1)

    if(col<3):
        col+=1
    else:
        col = 0
        row+=1

print("alpha = ", alpha_table)
print("dc = ", dc_table)

### Contour plots

y = np.arange(3,-1,-1)
x = np.arange(0,4,1)
X, Y = np.meshgrid(x, y)

### Contour plots
# First plot
fig3 = plt.figure()
Vax = plt.subplot2grid((1,1), (0,0))

# interpolation
#sigma = 0.5
#data1 = gaussian_filter(alpha_table, sigma)
data1 = alpha_table

#levels1 = [30.0,40.0,50.0,60.0,70.0]
levels1 = [5.0,10.0,15.0,20.0,25.0,30.0]
CS1 = Vax.contourf(X, Y, data1, levels1, cmap=plt.cm.Blues)

plt.setp(Vax, yticks=[])
plt.setp(Vax, xticks=[])

#cbar1 = plt.colorbar(CS1,ticks=[])
#cbar1 = plt.colorbar(CS1)

# Second plot
fig4 = plt.figure()
Gax = plt.subplot2grid((1,1), (0,0))

# interpolation
#sigma = 0.5
#data2 = gaussian_filter(dc_table, sigma)
data2 = dc_table

#levels2 = [1.8,1.9,2.0]
levels2 = [2.0,2.1,2.2,2.4,2.5]
CS2 = Gax.contourf(X, Y, data2, levels2, cmap=plt.cm.Blues)

plt.setp(Gax, yticks=[])
plt.setp(Gax, xticks=[])

#cbar2 = plt.colorbar(CS2,ticks=[])
#cbar2 = plt.colorbar(CS2)

###

plt.show()
