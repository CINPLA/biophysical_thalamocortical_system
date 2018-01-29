#!/usr/bin/env python
# -*- coding: utf-8 -*-

## Simulation of the retinal ganglion cells' response. Spike trains from ganglion
## cells are produced by Poisson generators in NEST with firing rates
## determined by a response function R_g(r, t). The response function R_g (r, t)
## is defined as a non-separable center-surround filter implemented by means of
## recursive linear fiters.

# MPI
from mpi4py import MPI
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

import numpy as np
from scipy.special import i0e
from scipy.integrate import quadrature
from scipy.integrate import dblquad
from scipy.special import erf
import os
import sys
import time
import nest
import matplotlib.pyplot as plt

#import dataPlot
#reload(dataPlot)

# Path to folder 'thalamocortical'
root_path = os.path.dirname(os.path.realpath(__file__))+"/"

#############################
#### Start of parameters ####
#############################

# Number of neurons per row (total number of neurons in the grid: N*N)
N = 10.0

# Visual angle (deg)
visSize = 10.0

# Stimulus type (0 = flashing circular spot, 1 = patch grating, 2 = receptive
# field, 3 = temporal impulse response, 4 = horizontal bar)
stimulustype = 0

# Grating parameters
Cpg = 0.7 # contrast
kpg = (2*np.pi)* 0.15 # spatial frequency (cpd)
thetakpg = 0.0 # drift angle
wpg = (2.0*np.pi)* 1.0 # temporal frequency (Hz)

# Bar parameters
# bard = 3.0

# Simulation parameters
tsim = 1000.0 # simulation time (ms)
tstart = 500.0 # onset of the stimulus (ms)
tstop = 140.0 # offset (only for the receptive field)
dt = 1.0 # time step (ms)
binsize = 5.0 # bin size used for PSTHs
numbertrials = 5.0 # repetitions of the experiment

# Select between ON/OFF ganglion cells
ganglion_cell_type = "ON"
# Type of spot (only for the receptive field). RF_1 = white spot, RF_2 =
# black spot
spot_type = "RF_1"

#############################
##### End of parameters #####
#############################

# Create stimulus
if stimulustype <= 1 or stimulustype == 4:
    # Range of spot/patch diameters (deg)
    stimulus = np.arange(0.0,10.2,0.2)

elif stimulustype == 2:
    # Receptive field
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

elif stimulustype == 3:
    # Temporal impulse response
    stimulus = [10.0**6]

# Folder to save results
if stimulustype == 0:
    type = "disk/"+ganglion_cell_type
elif stimulustype == 1:
    type = "patch_grating/0/"+ganglion_cell_type
elif stimulustype == 2:
    type = spot_type+"/"+ganglion_cell_type
elif stimulustype == 3:
    type = "impulse/"+ganglion_cell_type
elif stimulustype == 4:
    type = "bar/"+ganglion_cell_type

# Create folders
if rank == 0:
    os.system("mkdir "+root_path+"results/retina")
    if stimulustype == 0:
        os.system("mkdir "+root_path+"results/retina/disk")
        os.system("mkdir "+root_path+"results/retina/disk/"+ganglion_cell_type)
    elif stimulustype == 1:
        os.system("mkdir "+root_path+"results/retina/patch_grating")
        os.system("mkdir "+root_path+"results/retina/patch_grating/0")
        os.system("mkdir "+root_path+"results/retina/patch_grating/0/"+ganglion_cell_type)
    elif stimulustype == 2:
        os.system("mkdir "+root_path+"results/retina/"+spot_type)
        os.system("mkdir "+root_path+"results/retina/"+spot_type+"/"+ganglion_cell_type)
    elif stimulustype == 3:
        os.system("mkdir "+root_path+"results/retina/impulse")
        os.system("mkdir "+root_path+"results/retina/impulse/"+ganglion_cell_type)
    elif stimulustype == 4:
        os.system("mkdir "+root_path+"results/retina/bar")
        os.system("mkdir "+root_path+"results/retina/bar/"+ganglion_cell_type)

#############################
######## MPI function #######
#############################

# Each worker runs simulations of the allocated subset of stimuli
def worker(stimulus):

    PST_copy = []

    for stim in stimulus:
        for trial in np.arange(0,numbertrials):

            print("Trial = %s, Stimulus = %s" % (trial,stim))

            # new object Retina
            retina = Retina()
            # Initialize PSTHs at the first trial
            if (trial == 0):
                retina.PSTinit(tsim,binsize,N)
                response = np.zeros((int(N*N),int(tsim/dt)))
            else:
                retina.PST_avg = PST_copy

            # Set the center of the grid
            if(stimulustype < 4):
                retina.stimCenter = [N/2,N/2] # Disk/grating
            else:
                retina.stimCenter = [0.0,N/2] # Horizontal bar

            # Create cells
            for n in np.arange(0,N*N):
                if stimulustype==1:
                    retina.create_cell((0.0,0.0,0.85,750.0,850.0,0.62,1.26,dt)) # Grating
                elif stimulustype==2:
                    retina.create_cell((0.0,0.0,0.85,600.0,900.0,0.62,1.26,dt)) # Spot, 50 % contrast
                else:
                    retina.create_cell((0.0,0.0,0.85,245.0,377.0,0.62,1.26,dt)) # Disk

            # Retina initialization functions
            retina.updatePosition(N,visSize)
            retina.create_nest_objects()
            retina.resetValues()

            # Simulation
            print("simulation")

            for t in np.arange(0.0,tsim,retina.cells[0].dt):
                if stimulustype == 0:
                    response = retina.update_disk(stim,t,tstart,trial,response)
                elif stimulustype == 1:
                    response = retina.update_grating(stim,Cpg,kpg,thetakpg,wpg*(-t+500.0)*0.001,t,tstart,trial,response)
                elif stimulustype == 2:
                    response = retina.update_spatial_impulse(t,tstart,tstop,trial,response,stim,N)
                elif stimulustype == 3:
                    response = retina.update_temporal_impulse(t,trial,response)
                elif stimulustype == 4:
                    response = retina.update_bar(bard,stim,t,tstart,trial,response)

                retina.simulate(retina.cells[0].dt)

                # move bar
                if stimulustype == 4:
                    if(np.remainder(t,10.0) == 0.0): # move every 10 ms
                        retina.move_bar(0.1,0.0,N,visSize)

            # Get spike times
            print("analysis")
            retina.analysis()
            # Update PSTHs
            retina.PSTupdate(trial,stim,tsim,binsize,type,response)
            PST_copy = retina.PST_avg

            # checkpoint (plot response)
#            plt.plot(np.arange(0.0,tsim,retina.cells[0].dt),response[55,:])
#            plt.xlabel('time(ms)')
#            plt.ylabel('GC Response')
#            plt.show()

        # Save PSTH matrix
        retina.savePST(stim,type)

    return 1


## Linear temporal filter (recursive implementation: IIR filters)
class LinearFilter(object):

    def __init__(self,tau,n,step):
        self.tau = tau
        self.step = step
        self.M = 1
        self.N = n + 1

        if(n):
            tauC = self.tau/n
        else:
            tauC = self.tau

        c = np.exp(-self.step/tauC)
        self.b = np.zeros(1)
        self.b[0] = np.power(1-c,self.N)
        self.a = np.zeros(self.N+1)

        for i in np.arange(0,self.N+1,1):
            self.a[i] = np.power(-c,i) * self.combination(self.N,i)

        self.last_inputs = np.zeros(self.M)
        self.last_values = np.zeros(self.N+1)

    # Combinatorials of gamma function:
    def arrangement(self,n,k):
        res=1.0
        for i in np.arange(n,n-k,-1):
            res*=i

        return res

    def combination(self,n,k):
        return self.arrangement(n,k)/self.arrangement(k,k)

    # New input
    def feedInput(self,new_input):
        self.last_inputs[0]=new_input

    # Update filter coefficients
    def update(self):

        # Rotation on addresses of the last_values.
        fakepoint=self.last_values[self.N]
        for i in np.arange(1,self.N+1,1):
            self.last_values[self.N+1-i]=self.last_values[self.N-i]
        self.last_values[0]=fakepoint

        # Calculating new value of filter recursively:
        self.last_values[0] = self.b[0]* self.last_inputs[0]
        for j in np.arange(1,self.M,1):
            self.last_values[0] += ( self.b[j] * self.last_inputs[j] )
        for k in np.arange(1,self.N+1,1):
            self.last_values[0] -= ( self.a[k] * self.last_values[k] )
        if(self.a[0]!=1.0):
            self.last_values[0] = self.last_values[0] / self.a[0]

    # reset values
    def resetValues(self):
        self.last_inputs = np.zeros(self.M)
        self.last_values = np.zeros(self.N+1)

## Ganglion cell
class ganglionCell(object):

    def __init__(self, g, thetag, w, l_bkg, l_stim, a1, a2, dt):

        # parameters of the Gaussian filtering
        self.g = g
        self.thetag = thetag
        self.w = w
        self.l_bkg = l_bkg
        self.l_stim = l_stim
        self.a1 = a1
        self.a2 = a2

        # temporal IIR filters
        self.dt = dt # ms
        self.tau_c_lp = 20.0 # ms
        self.tau_c_hp = 50.0 # ms (between 30 and 50 ms)
        self.tau_s = 50.0 # ms

        self.alpha = 1.0
        self.beta = 2.0

        self.IIR_lp = LinearFilter(self.tau_c_lp,4.0,self.dt)
        self.IIR_hp = LinearFilter(self.tau_c_hp,1.0,self.dt)
        self.IIR_s = LinearFilter(self.tau_s,5.0,self.dt)

        # to save NEST spikes
        self.spikes = []

    # Spatial convolution with a grating (d = patch diameter, a = spatial extent
    # of the kernel, g = distance from the center grid, thetag = angle of the
    # position vector, Cpg, kpg, thetakpg, wpgt = grating parameters)
    def spatial_conv_grating(self,d,a,g,thetag,Cpg,kpg,thetakpg,wpgt):

#        acc = 0.0
#        delta = 30.0

#        deltar = (d/2.0) / delta
#        deltatheta = (2.0*np.pi) / delta

#        # Approximation
#        for r in np.arange(0.0,(d/2.0) + deltar,deltar):
#            for theta in np.arange(0.0,(2.0*np.pi) + deltatheta,deltatheta):
#                acc+= (deltar*deltatheta) * (1.0/(np.pi * a**2)) * np.exp(-g**2/a**2) * np.exp(-r**2/a**2) *\
#                    np.exp((2.0*g*r*np.cos(theta-thetag))/a**2)*r*\
#                    Cpg*np.cos(kpg*r*np.cos(theta-thetakpg) - wpgt)

#        return acc

        # Exact integration
        return dblquad(lambda theta, r: (1.0/(np.pi * a**2)) * np.exp(-g**2/a**2) * np.exp(-r**2/a**2) *\
                                        np.exp((2.0*g*r*np.cos(theta-thetag))/a**2)*r*\
                                        Cpg*np.cos(kpg*r*np.cos(theta-thetakpg) - wpgt)  ,
                        0.0, d/2.0, lambda x:0.0, lambda x:2.0*np.pi)[0]

    # Spatial convolution with a bar (d = patch diameter, l = bar length, a = spatial
    # extent of the kernel, g = distance from the center grid, thetag = angle of the
    # position vector)
    def spatial_conv_bar(self,d,l,a,g,thetag):
        gx = g * np.cos(thetag)
        gy = g * np.sin(thetag)

        return (1.0/(np.pi * a**2)) * np.exp(-(gx**2 + gy**2)/(a**2))*\
        0.5*np.sqrt(np.pi)*a*np.exp(gx**2/a**2)*(erf((d/2.0-gx)/a) - erf((-d/2.0-gx)/a))*\
        0.5*np.sqrt(np.pi)*a*np.exp(gy**2/a**2)*(erf((l/2.0-gy)/a) - erf((-l/2.0-gy)/a))

    # Spatial convolution with a spot (d = patch diameter, g = distance from the
    # center grid, a = spatial extent of the kernel)
    def spatial_conv_disk(self, d, g, a, short_circuit=False):

        g0 = g / a

        # cell at the center of the grid
        if short_circuit:
            if g == 0.:
                return 1.-np.exp(-(d/(2*a))**2)

        def integrand(r0):
            return 2.*r0*np.exp(-(r0-g0)**2)*i0e(2.*r0*g0)
            #return 2.*r0*np.exp(-r0**2 - g0**2)*i0e(2.*r0*g0)

        return quadrature(integrand, 0, d/(2.*a))[0]


    # Ganglion cell's response to a moving bar
    def total_conv_bar(self, d,l,t,tstart):
        c = 0.0
        s = 0.0

        if(t<tstart):
            c = 0.0
            s = 0.0
        else:
            # Center and surround signals
            c = (self.l_stim - self.l_bkg) * self.spatial_conv_bar(d,l,self.a1, self.g, self.thetag)
            s = (self.l_stim - self.l_bkg) *self.w * self.spatial_conv_bar(d,l,self.a2, self.g, self.thetag)

        # Temporal convolution
        self.IIR_lp.feedInput(self.beta*c - self.IIR_hp.last_values[0])
        self.IIR_s.feedInput(s)
        self.IIR_hp.feedInput(c)
        self.IIR_lp.update()
        self.IIR_s.update()
        self.IIR_hp.update()

        # ON/OFF responses
        if ganglion_cell_type == "ON":
            return self.alpha*(self.IIR_lp.last_values[0] - self.IIR_s.last_values[0]) +\
                self.l_bkg*(1.0-self.w)
        else:
            return self.alpha*(-self.IIR_lp.last_values[0] + self.IIR_s.last_values[0]) +\
                self.l_bkg*(1.0-self.w)

    # Ganglion cell's response to a drifting sinusoidal grating
    def total_conv_grating(self, d,Cpg,kpg,thetakpg,wpgt,t,tstart):
        c = 0.0
        s = 0.0

        if(t<tstart):
            c = 0.0
            s = 0.0
        else:
            # Center and surround signals
            c = (self.l_stim - self.l_bkg) * self.spatial_conv_grating(d,self.a1,self.g,self.thetag,Cpg,kpg,thetakpg,wpgt)
            s = (self.l_stim - self.l_bkg) *self.w * self.spatial_conv_grating(d,self.a2,self.g,self.thetag,Cpg,kpg,thetakpg,wpgt)

        # Temporal convolution
        self.IIR_lp.feedInput(self.beta*c - self.IIR_hp.last_values[0])
        self.IIR_s.feedInput(s)
        self.IIR_hp.feedInput(c)
        self.IIR_lp.update()
        self.IIR_s.update()
        self.IIR_hp.update()

        # ON/OFF responses
        if ganglion_cell_type == "ON":
            return self.alpha*(self.IIR_lp.last_values[0] - self.IIR_s.last_values[0]) +\
                self.l_bkg*(1.0-self.w)
        else:
            return self.alpha*(-self.IIR_lp.last_values[0] + self.IIR_s.last_values[0]) +\
                self.l_bkg*(1.0-self.w)

    # Ganglion cell's response to a flashing spot
    def total_conv_disk(self, d,t,tstart):
        c = 0.0
        s = 0.0

        if(t<tstart):
            c = 0.0
            s = 0.0
        else:
            # Center and surround signals
            c = (self.l_stim - self.l_bkg) * self.spatial_conv_disk(d, self.g, self.a1)
            s = (self.l_stim - self.l_bkg) *self.w * self.spatial_conv_disk(d, self.g, self.a2)

        # Temporal convolution
        self.IIR_lp.feedInput(self.beta*c - self.IIR_hp.last_values[0])
        self.IIR_s.feedInput(s)
        self.IIR_hp.feedInput(c)
        self.IIR_lp.update()
        self.IIR_s.update()
        self.IIR_hp.update()

        # ON/OFF responses
        if ganglion_cell_type == "ON":
            return self.alpha*(self.IIR_lp.last_values[0] - self.IIR_s.last_values[0]) +\
                self.l_bkg*(1.0-self.w)
        else:
            return self.alpha*(-self.IIR_lp.last_values[0] + self.IIR_s.last_values[0]) +\
                self.l_bkg*(1.0-self.w)

    # Ganglion cell's temporal impulse response
    def total_conv_temporal_impulse(self,t):
        c = 0.0
        s = 0.0

        if(t>100.0 and t<=100.0+self.dt):
            # Center and surround signals (d is sufficiently big to cover the whole RF)
            c = self.spatial_conv_disk(10.0*self.a2, self.g, self.a1)
            s = self.w * self.spatial_conv_disk(10.0*self.a2, self.g, self.a2)
        else:
            c =  0.0
            s =  0.0

        # Temporal convolution
        self.IIR_lp.feedInput(self.beta*c - self.IIR_hp.last_values[0])
        self.IIR_s.feedInput(s)
        self.IIR_hp.feedInput(c)
        self.IIR_lp.update()
        self.IIR_s.update()
        self.IIR_hp.update()

        # ON/OFF responses
        if ganglion_cell_type == "ON":
            return self.alpha*(self.IIR_lp.last_values[0] - self.IIR_s.last_values[0]) +\
                self.l_bkg*(1.0-self.w)
        else:
            return self.alpha*(-self.IIR_lp.last_values[0] + self.IIR_s.last_values[0]) +\
                self.l_bkg*(1.0-self.w)

    # Ganglion cell's receptive field
    def total_conv_spatial_impulse(self, d,t,tstart,tstop,r):
        c = 0.0
        s = 0.0

        if(t<tstart or t>tstop):
            c = 0.0
            s = 0.0
        else:
            # Center and surround signals
            # White spot
            if spot_type == "RF_1":
                c = (self.l_stim - self.l_bkg) * self.spatial_conv_disk(d, r, self.a1)
                s = (self.l_stim - self.l_bkg) *self.w * self.spatial_conv_disk(d, r, self.a2)
            # Black spot
            else:
                c = (-self.l_stim + self.l_bkg) * self.spatial_conv_disk(d, r, self.a1)
                s = (-self.l_stim + self.l_bkg) *self.w * self.spatial_conv_disk(d, r, self.a2)

        # Temporal convolution
        self.IIR_lp.feedInput(self.beta*c - self.IIR_hp.last_values[0])
        self.IIR_s.feedInput(s)
        self.IIR_hp.feedInput(c)
        self.IIR_lp.update()
        self.IIR_s.update()
        self.IIR_hp.update()

        # ON/OFF responses
        if ganglion_cell_type == "ON":
            return self.alpha*(self.IIR_lp.last_values[0] - self.IIR_s.last_values[0]) +\
                self.l_bkg*(1.0-self.w)
        else:
            return self.alpha*(-self.IIR_lp.last_values[0] + self.IIR_s.last_values[0]) +\
                self.l_bkg*(1.0-self.w)

    # Reset filter coefficients
    def resetValues(self):
        self.IIR_lp.resetValues()
        self.IIR_hp.resetValues()
        self.IIR_s.resetValues()


## Retina
class Retina(object):

    def __init__(self):
        self.cells = [] # array of ganglion cells
        self.generators = [] # array of spike generators
        self.spikedetector = 0 # spike detector in NEST
        self.stimCenter = [0,0] # center of the grid
        self.PST_avg = [] # accumulator of PSTHs
        self.poisson = True # True = true random generators for every trial,
                            # False = same seeds for every trial

    # Add a new ganglion cell
    def create_cell(self, cell_parameters):
        self.cells.append(ganglionCell(*cell_parameters))

    # Create and initialize NEST objects
    def create_nest_objects(self):

        # Number of NEST threads
        NESTthr = 1

        # Simulation parameters
        sd_params = {'to_memory': True, 'withgid': True, 'withtime': True}

        # Initialize seeds of random generators
        if(self.poisson):
            np.random.seed(int(time.time()))
            seeds = np.arange(NESTthr) + int((time.time()*100)%2**32)
        else:
            seeds = np.arange(NESTthr)

        # Initialize NEST kernel
        nest.ResetKernel()
        nest.ResetNetwork()
        nest.SetKernelStatus({"local_num_threads": NESTthr,"resolution": self.cells[0].dt, "rng_seeds": list(seeds)})
        nest.set_verbosity('M_QUIET')

        self.generators = []
        self.spikedetector = 0

        generator_params = {'rate' : 10.}
        # Create Poisson generators
        for cell in self.cells:
            generator = nest.Create('poisson_generator', n = 1, params = generator_params)
            self.generators.append(generator[0])

        # Spike detector
        self.spikedetector = nest.Create('spike_detector', params = sd_params)
        nest.SetStatus(self.spikedetector, [{"n_events": 0}])
        nest.Connect(self.generators, self.spikedetector)

    # Compute a new retina response to the flashing spot
    def update_disk(self,dd,t,tstart,trial,response):

        cell_number = 0
        for cell in self.cells:
            if trial==0:
                rate = self.cells[cell_number].total_conv_disk(dd,t,tstart)
                response[cell_number,int(t/self.cells[0].dt)] = rate
            else:
                rate = response[cell_number,int(t/self.cells[0].dt)]

            if(rate>= 0):
                nest.SetStatus([self.generators[cell_number]],{'rate':rate})
            else:
                nest.SetStatus([self.generators[cell_number]],{'rate':0.0})

            cell_number += 1

        return response

    # Compute a new retina response to the moving bar
    def update_bar(self,dd,ll,t,tstart,trial,response):

        cell_number = 0
        for cell in self.cells:
            if trial==0:
                rate = self.cells[cell_number].total_conv_bar(dd,ll,t,tstart)
                response[cell_number,int(t/self.cells[0].dt)] = rate
            else:
                rate = response[cell_number,int(t/self.cells[0].dt)]

            if(rate>= 0):
                nest.SetStatus([self.generators[cell_number]],{'rate':rate})
            else:
                nest.SetStatus([self.generators[cell_number]],{'rate':0.0})

            cell_number += 1

        return response

    # Shift the bar
    def move_bar(self,dx,dy,N,visSize):

        cell_number = 0
        self.stimCenter[0]+= dx
        self.stimCenter[1]+= dy
        self.updatePosition(N,visSize)

    # Compute a new retina response to the grating
    def update_grating(self,dd,Cpg,kpg,thetakpg,wpgt,t,tstart,trial,response):

        cell_number = 0
        for cell in self.cells:
            if trial==0:
                rate = self.cells[cell_number].total_conv_grating( dd,Cpg,kpg,thetakpg,wpgt,t,tstart)
                response[cell_number,int(t/self.cells[0].dt)] = rate
            else:
                rate = response[cell_number,int(t/self.cells[0].dt)]

            if(rate>= 0):
                nest.SetStatus([self.generators[cell_number]],{'rate':rate})
            else:
                nest.SetStatus([self.generators[cell_number]],{'rate':0.0})

            cell_number += 1

        return response

    # Compute a new retina response for the impulse response
    def update_temporal_impulse(self,t,trial,response):

        cell_number = 0
        for cell in self.cells:
            if trial==0:
                rate = self.cells[cell_number].total_conv_temporal_impulse(t)
                response[cell_number,int(t/self.cells[0].dt)] = rate
            else:
                rate = response[cell_number,int(t/self.cells[0].dt)]

            if(rate>= 0):
                nest.SetStatus([self.generators[cell_number]],{'rate':rate})
            else:
                nest.SetStatus([self.generators[cell_number]],{'rate':0.0})

            cell_number += 1

        return response

    # Compute a new retina response for generation of the receptive field
    def update_spatial_impulse(self,t,tstart,tsop,trial,response,pos,N):

        cell_number = 0
        for cell in self.cells:
            if trial==0:
                r = np.sqrt((int(pos/Npoints)*(N/Npoints) - int(cell_number/N))**2 +\
                (np.remainder(pos,Npoints)*(N/Npoints) - np.remainder(cell_number,N))**2) *\
                (visSize/N)
                rate = self.cells[cell_number].total_conv_spatial_impulse(1.0,t,tstart,tstop,r)
                response[cell_number,int(t/self.cells[0].dt)] = rate
            else:
                rate = response[cell_number,int(t/self.cells[0].dt)]

            if(rate>= 0):
                nest.SetStatus([self.generators[cell_number]],{'rate':rate})
            else:
                nest.SetStatus([self.generators[cell_number]],{'rate':0.0})

            cell_number += 1

        return response

    # Create position vectors of ganglion cells
    def updatePosition(self,N,visSize):

        cell_number = 0
        for x in np.arange(0,N):
            for y in np.arange(0,N):

                module = np.sqrt((x-self.stimCenter[0])**2 + (y-self.stimCenter[1])**2) * (visSize/N)
                if(x== self.stimCenter[0] and y==self.stimCenter[1]):
                    angle = 0.0
                else:
                    angle = np.arccos((x-self.stimCenter[0])/( np.sqrt((x-self.stimCenter[0])**2 + (y-self.stimCenter[1])**2)) )

                self.cells[cell_number].g = module
                self.cells[cell_number].thetag = angle

                cell_number += 1

    # NEST simulation
    def simulate(self, tsim):
        nest.Simulate(tsim)

    # Reset coefficients of ganglion cells
    def resetValues(self):
        for cell in self.cells:
            cell.resetValues()

    # PSTH initialization
    def PSTinit(self,tsim,binsize,N):
        self.PST_avg = np.zeros((int(N*N),int(tsim/binsize)))

    # Count spikes
    def analysis(self):
        spikes = nest.GetStatus(self.spikedetector, 'events')[0]
        senders = spikes['senders']
        times = spikes['times']

        cell_number = 0
        for cell in self.cells:
            selected_senders = np.where(senders==self.generators[cell_number])
            cell.spikes = times[selected_senders[0]]
            cell_number += 1

    # Update PSTH
    def PSTupdate(self,trial,stim,tsim,binsize,type,retina):

        # to save spikes
        os.system("mkdir "+root_path+"results/retina/"+type+"/stim"+str(stim))
        text_file = open(root_path+"results/retina/"+type+"/stim"+str(stim)+"/spikes"+str(trial), "w")

        # get spikes
        spikes = []
        for n in np.arange(0,len(self.cells)):
            spikes.append(np.array(self.cells[int(n)].spikes))

            # Save spikes to file
            for ch in spikes[n]:
                text_file.write(str(ch))
                text_file.write(",")
            text_file.write(os.linesep)

            # update PST matrix
            # A) Standard method
            if(self.poisson):
                h, e = np.histogram(spikes[n], bins=np.arange(0., tsim+binsize, binsize))
                self.PST_avg[n,:] += h * (1000. / binsize)
            # B) fake method
            else:
                binsize2 = 100.0 # ms
                spikesnp = np.array(spikes[n])
                for t in np.arange(0.0,tsim-binsize2,binsize):
                    first = spikesnp[np.where(spikesnp>=t)[0]]
                    second = first[np.where(first< t+binsize2 )[0]]
                    self.PST_avg[n,int((t + binsize2/2.0)/binsize)] = len(second) * (1000. / binsize2)

        text_file.close()

    # Save PSTH to file
    def savePST(self,stim,type):

        text_file = open(root_path+"results/retina/"+type+"/stim"+str(stim)+"/PST", "w")

        for line in np.arange(0,len(self.PST_avg)):

            for ch in self.PST_avg[line,:]:
                text_file.write(str(ch))
                text_file.write(",")
            text_file.write(os.linesep)

        text_file.close()

    # Load PSTH from file
    def loadPST(self,stim,N,tsim,binsize,type):

        PST_avg = np.zeros((int(N*N),int(tsim/binsize)))

        lines = [line.rstrip('\n') for line in open(root_path+"results/retina/"+type+"/stim"+str(stim)+"/PST", "r")]

        for n in np.arange(N*N):
            h = lines[int(n)].split(',')
            for pos in np.arange(0,tsim/binsize):
                PST_avg[int(n),int(pos)] = float(h[int(pos)])

        return PST_avg

    # Load spikes from file
    def loadSpikes(self,stim,trial,N,type):

        spikes = []

        lines = [line.rstrip('\n') for line in open(root_path+"results/retina/"+type+"/stim"+str(stim)+"/spikes"+str(trial), "r")]

        for n in np.arange(N*N):
            h = lines[int(n)].split(',')
            sp = []
            for pos in np.arange(0,len(h)-1):
                sp.append( float(h[pos]) )
            spikes.append(sp)

        return spikes



#############################
##### ! Main ################
#############################

def main():

    # Start timer
    if rank==0:
        start_c = time.time()

    # Divide data into chunks
    if rank == 0:
        chunks = [[] for _ in range(size)]
        for i, chunk in enumerate(stimulus):
            chunks[i % size].append(chunk)
    else:
        chunks = None


    # Scatter data
    stim = []
    stim = comm.scatter(chunks,root=0)
    value_to_return = worker(stim)
    # Gather data
    results = comm.gather(value_to_return, root=0)

    # Show time elapsed
    if rank == 0:
        end_c = time.time()
        print("time elapsed (h): %s" % str((end_c - start_c)/3600.0))

if __name__ == '__main__':
     main()


################################################################################################
####### Old method to compute integral of bar ##################################################
################################################################################################

#from scipy.integrate import dblquad
#    def spatial_conv_bar(self,d,l,a,g,thetag):

#        # Corrected angle
#        def corrAngle(y,x):
#            if(x != 0.0):
#                if ((x>= 0.0 and y>=0.0) or (x>=0.0 and y<0.0)):
#                    return np.arctan(y/x)
#                elif ((x<0.0 and y>=0.0) or (x<0.0 and y<0.0)):
#                    return np.arctan(y/x) + np.pi
#            else:
#                return np.sign(y)*(np.pi/2)

#        ### double integral with dblquad
#    #    def integrand(r,theta):
#    #        return (1.0/(np.pi*a**2)) * np.exp(-r**2/a**2) * np.exp(-g**2/a**2) *np.exp(2*r*g*np.cos(theta - thetag)/a**2) * r

#    #    return dblquad(integrand, corrAngle(-l,d), corrAngle(l,d), lambda x: 0, lambda x: (d/2.0)/np.cos(x))[0] +\
#    #    dblquad(integrand, corrAngle(l,d), corrAngle(l,-d), lambda x: 0, lambda x: (l/2.0)/np.sin(x))[0] +\
#    #    dblquad(integrand, corrAngle(l,-d), corrAngle(-l,-d), lambda x: 0, lambda x: (d/2.0)/np.cos(x))[0] +\
#    #    dblquad(integrand, corrAngle(-l,-d) - 2*np.pi, corrAngle(-l,d), lambda x: 0, lambda x: (l/2.0)/np.sin(x))[0]

#        def func(theta,r):
#            cc = np.cos(thetag-theta)
#            return  (1.0/(np.pi*a**2)) * np.exp(-g**2/a**2) * 0.5 *\
#                    ( np.sqrt(np.pi)*a*g*cc*np.exp(-((g**2)*(-np.cos(2*(theta-thetag))) - g**2) / (2*a**2)) * \
#                    (erf((r - g*cc)/a) - erf((-g*cc)/a)) - (a**2)*(np.exp((r/a**2)*(2*g*cc - r)) - 1) )

#        def integrand1(theta):
#            r = (d/2.0) / np.abs(np.cos(theta))
#            return func(theta,r)

#        def integrand2(theta):
#            r = (l/2.0) / np.abs(np.sin(theta))
#            return func(theta,r)


#        return quadrature(integrand1, corrAngle(-l,d), corrAngle(l,d))[0] +\
#                quadrature(integrand2, corrAngle(l,d), corrAngle(l,-d))[0] +\
#                quadrature(integrand1, corrAngle(l,-d), corrAngle(-l,-d))[0] +\
#                quadrature(integrand2, corrAngle(-l,-d) - 2*np.pi, corrAngle(-l,d))[0]
