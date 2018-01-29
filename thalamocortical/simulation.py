#!/usr/bin/env python
# -*- coding: utf-8 -*-

## This class provides functions to run the simulation in NEURON and to save/load
## PSTHs and membrane potentials from neurons.

from os.path import join
import os
import numpy as np
import pylab as plt
from time import time
import neuron
import LFPy
import os

nrn = neuron.h

# Path to folder 'thalamocortical'
root_path = os.path.dirname(os.path.realpath(__file__))+"/"

class Simulation(object):

    def __init__(self):
        print("init")


    def simulateCells(self,cells):

        # set recorders
        for c in cells:
            self._set_soma_volt_recorder(c)
            self._collect_tvec(c)
            self._set_voltage_recorders(c)

        # simulate
        self._run_simulation(cells, False, atol=0.001)

        #somatic trace
        for c in cells:
            c.somav = np.array(c.somav)
            self._collect_vmem(c)


    def _run_simulation(self,cells, variable_dt=False, atol=0.001):
        '''
        Running the actual simulation in NEURON, simulations in NEURON
        is now interruptable.
        '''
        nrn.dt = cells[0].timeres_NEURON

        cvode = nrn.CVode()

        #don't know if this is the way to do, but needed for variable dt method
        if variable_dt:
            cvode.active(1)
            cvode.atol(atol)
        else:
            cvode.active(0)

        #initialize state
        for c in cells:
            nrn.finitialize(c.v_init)

        #initialize current- and record
        if cvode.active():
            cvode.re_init()
        else:
            nrn.fcurrent()
        nrn.frecord_init()

        ##Starting simulation at tstart
        nrn.t = cells[0].tstartms

        for c in cells:
            c._loadspikes()

        #print sim.time and realtime factor at intervals
        counter = 0.
        t0 = time()
        ti = nrn.t
        if cells[0].tstopms > 10000:
            interval = 1 / cells[0].timeres_NEURON * 1000
        else:
            interval = 1 / cells[0].timeres_NEURON * 100

        while nrn.t < cells[0].tstopms:
            nrn.fadvance()
            counter += 1.
            if np.mod(counter, interval) == 0:
                rtfactor = (nrn.t - ti)  * 1E-3 / (time() - t0)
                if cells[0].verbose:
                    print(('t = %.0f, realtime factor: %.3f' % (nrn.t, rtfactor)))
                t0 = time()
                ti = nrn.t



    def _set_soma_volt_recorder(self,cell):
        '''
        Record somatic membrane potential
        '''
        cell.somav = nrn.Vector(int(cell.tstopms /
                                         cell.timeres_python+1))
        if cell.nsomasec == 0:
            pass
        elif cell.nsomasec == 1:
            for sec in cell.somalist:
                cell.somav.record(sec(0.5)._ref_v,
                              cell.timeres_python)
        elif cell.nsomasec > 1:
            nseg = cell.get_idx('soma').size
            i, j = divmod(nseg, 2)
            k = 1
            for sec in cell.somalist:
                for seg in sec:
                    if nseg==2 and k == 1:
                        #if 2 segments, record from the first one:
                        cell.somav.record(seg._ref_v, cell.timeres_python)
                    else:
                        if k == i*2:
                            #record from one of the middle segments:
                            cell.somav.record(seg._ref_v,
                                              cell.timeres_python)
                    k += 1

    def _collect_tvec(self,cell):
        '''
        Set the tvec to be a monotonically increasing numpy array after sim.
        '''
        cell.tvec = np.arange(cell.tstopms / cell.timeres_python + 1) \
                            * cell.timeres_python

    def _set_voltage_recorders(self,cell):
        '''
        Record membrane potentials for all segments
        '''
        cell.memvreclist = nrn.List()
        for sec in cell.allseclist:
            for seg in sec:
                memvrec = nrn.Vector(int(cell.tstopms /
                                              cell.timeres_python+1))
                memvrec.record(seg._ref_v, cell.timeres_python)
                cell.memvreclist.append(memvrec)

    def _collect_vmem(self,cell):
        '''
        Get the membrane currents
        '''
        cell.vmem = np.array(cell.memvreclist)
        cell.memvreclist = None
        #del cell.memvreclist


    def deleteAll(self,cells):
        for c in cells:
            del c.somav
            del c.tvec
            del c.memvreclist
            del c.vmem

    # save PSTH
    def savePST(self,stim,neuron,PST,type):

        os.system("mkdir "+root_path+"results/thalamocortical/"+type)
        os.system("mkdir "+root_path+"results/thalamocortical/"+type+"/stim"+str(stim))
        text_file = open(root_path+"results/thalamocortical/"+type+"/stim"+str(stim)+"/PST"+neuron, "w")

        for line in np.arange(0,len(PST)):

            for ch in PST[line,:]:
                text_file.write(str(ch))
                text_file.write(",")
            text_file.write(os.linesep)

        text_file.close()

    # load PSTH
    def loadPST(self,stim,N,tsim,binsize,neuron,type):

        PST_avg = np.zeros((int(N*N),int(tsim/binsize)))

        lines = [line.rstrip('\n') for line in open(root_path+"results/thalamocortical/"+type+"/stim"+str(stim)+"/PST"+neuron, "r")]
        for n in np.arange(len(lines)):
            h = lines[int(n)].split(',')
            for pos in np.arange(0,tsim/binsize):
                PST_avg[int(n),int(pos)] = float(h[int(pos)])

        return PST_avg

    # Save membrane potential
    def saveMemPotential(self,stim,times,potentials,spikes_arriving,labels):

        os.system("mkdir "+root_path+"results/mem_potentials")
        self.saveArray(times,root_path+"results/mem_potentials/times"+str(stim),stim)
        self.saveArray(potentials,root_path+"results/mem_potentials/potentials"+str(stim),stim)
        self.saveArray(spikes_arriving,root_path+"results/mem_potentials/spikes_arriving"+str(stim),stim)
        self.saveString(labels,root_path+"results/mem_potentials/labels"+str(stim),stim)

    # Aux. functions
    def saveArray(self,arr,path,stim):

        text_file = open(path, "w")

        for line in np.arange(0,len(arr)):

            for ch in arr[line]:
                text_file.write(str(ch))
                text_file.write(",")
            text_file.write(os.linesep)

        text_file.close()

    def saveString(self,arr,path,stim):

        text_file = open(path, "w")

        for line in np.arange(0,len(arr)):

            for ch in arr[line]:
                text_file.write(str(ch))
            text_file.write(os.linesep)

        text_file.close()
