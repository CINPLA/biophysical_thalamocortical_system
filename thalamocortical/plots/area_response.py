#!/usr/bin/env python
# -*- coding: utf-8 -*-

## Area-response curves of the neurons located in the same position for all layers.

import numpy as np
import matplotlib.pyplot as plt
import sys,os,os.path
import scipy.fftpack

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

# Neuron type to plot
#IDs = ["RC-ON"]
IDs = [""] # Ganglion cells

# Simulation parameters
tsim = 1000.0
binsize = 5.0
numbertrials =5.0

# Interval to average spot response
spot_interval = [500.0,1000.0]

# Cell ID
cell_number = 55 # all layers except INs
cell_number_IN = 12

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

# Fourier fundamental frequency
def FFT(PST,cell_n,numbertrials):

    # Averaged response among trials
    response = PST[cell_n,:]/numbertrials
    # Number of samplepoints
    N = len(response)
    # Bin size and grating frequency
    resolution = binsize # ms
    temporal_frequency = 1.0 # Hz
    # Sample spacing
    T = resolution*0.001 # s

    # FFT
    yf = scipy.fftpack.fft(response)
    yf = 2.0/N * np.abs(yf[:N//2])
    xf = np.array(np.linspace(0.0, 1.0/(2.0*T), N/2))
    main_freq = np.where(xf>=temporal_frequency)[0][0]

    # Amplitude of F1
    return yf[main_freq]

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
            # FFT (amplitude of F1)
            # response.append(FFT(PST,cell_n,numbertrials) + sp_rate)

            # Mean firing rate of the rectified response over one cycle.
            # This metric is correlated with F1 (the average value of a sine wave
            # of voltage or current is 0.637 times the peak value)
#            PST = PST[cell_n,int(250.0/binsize):int(1250.0/binsize)]/numbertrials
#            response.append( np.mean(np.abs(PST - sp_rate)) + sp_rate)

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

    # To add d = 0
    xdata = np.concatenate((np.array([0.0]),np.array(xdata)))
    new_response = np.concatenate((np.array([response[0]]),new_response))

    return xdata,new_response


# Plots
fig = plt.figure()
fig.subplots_adjust(hspace=0.4)

Vax = plt.subplot2grid((2,1), (0,0))
Gax = plt.subplot2grid((2,1), (1,0))

for ID in IDs:

    newPST = createPST(ID,stimulus,N,tsim,binsize,folder)

    # Response
    if(ID == "IN-ON" or ID== "IN-OFF"):
        response = area_response(newPST,cell_number_IN)
    else:
        response = area_response(newPST,cell_number)

    # Interpolated response
    xdata,new_response = interpolation(response,stimulus)

    # Absolute response
    Vax.plot(xdata,new_response,label = ID)

    # Save to file
#     np.savetxt('/home/pablo/Desktop/data/patch_absolute.out',new_response,delimiter=',')

    # Calculate metrics
    print(ID)
    alphavr(new_response,xdata)

    # Normalized response
    new_response -= np.min(new_response)
    Gax.plot(xdata,new_response/np.max(new_response),label = ID)

    # Save to file
#    np.savetxt('/home/pablo/Desktop/data/patch_norm.out',
#    new_response/np.max(new_response),delimiter=',')

Vax.legend(loc=1)
Vax.set_ylabel('Firing rate (Hz)')
Gax.set_ylabel('Normalized firing rate')
Gax.set_xlabel('Spot/patch diameter (deg)')
plt.show()
