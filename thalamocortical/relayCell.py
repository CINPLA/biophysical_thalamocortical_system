#!/usr/bin/env python
# -*- coding: utf-8 -*-

## This class defines the properties and synapses of a dLGN relay-cell.

from os.path import join
import numpy as np
import pylab as plt
import neuron
import LFPy
import os

nrn = neuron.h

# Path to folder of neuron models
root_path = os.path.dirname(os.path.realpath(__file__))
root_path = root_path[0:root_path.find("thalamocortical")]

class RCTemplate(object):

    def __init__(self):

        self.RGC_SYN = dict(
            syntype='Exp2Syn',
            weight=0.0156,#0.014
            tau1=0.1,
            tau2=1.2,
            e=10.
        )

        self.LOCKED_SYN = dict(
            syntype='Exp2Syn',
            weight=0.004,
            tau1=.7,
            tau2=4.2,
            e=-80.
        )

    def return_cell(self):

        self.model_path = join(root_path+'TC_neuron')
        neuron.load_mechanisms(self.model_path)

        cell_parameters = {
            'morphology': join(self.model_path, 'soma.hoc'),
            'passive':False,
            'v_init' : -60,
            'extracellular':False,
            'nsegs_method': 'none',
            'timeres_NEURON':0.1,
            'timeres_python':0.1,
            'tstopms':2000,
            'tstartms':0,
            'templatefile': join(self.model_path, 'TC_GH2.hoc'),
            'templatename':'sTC',
            'templateargs':None,
            'delete_sections':False
        }

        cell = LFPy.TemplateCell(**cell_parameters)

        return cell


    def currentPulse(self,cell, stimamp0=0.055, stimamp1=0.0):

        PPparams0 = {
            'idx' : 0,
            'pptype' : 'IClamp',
            'delay' : 100,
            'dur' : 900,
            'amp' : stimamp0
        }
        PPparams1 = {
            'idx' : 0,
            'pptype' : 'IClamp',
            'delay' : 0,
            'dur' : 20000,
            'amp' : stimamp1,
        }

        if stimamp0 != 0:
            stim0 = LFPy.StimIntElectrode(cell, **PPparams0)
        if stimamp1 != 0:
            stim1 = LFPy.StimIntElectrode(cell, **PPparams1)

        cell.simulate(rec_vmem=True)


    def createSynapse(self, cell, location, locked, spikes, delay=1.0):

        if(locked == True):
            syn = LFPy.Synapse(cell, idx=location, **self.LOCKED_SYN)
            syn.set_spike_times(spikes+delay)
        else:
            syn = LFPy.Synapse(cell, idx=location, **self.RGC_SYN)
            syn.set_spike_times(spikes)

        cell._loadspikes()

        return syn


    def triadSynapse(self, cell):
        syn = nrn.Exp2Syn(0.5, sec=cell.cell.soma[0])
        syn.e = -80.0
        syn.tau1 = .45
        syn.tau2 = 5.0

        return syn


    def somaInhibition(self, cell):
        syn = nrn.Exp2Syn(0.5, sec=cell.cell.soma[0])
        syn.e = -60.0
        syn.tau1 = .45
        syn.tau2 = 5.0

        return syn

    def somaInhibitionGABAB(self, cell):
        syn = nrn.Exp2Syn(0.5, sec=cell.cell.soma[0])
        syn.e = -80.0
        syn.tau1 = 60.0
        syn.tau2 = 200.0

        return syn

    def somaExcitation(self, cell):
        syn = nrn.Exp2Syn(0.5, sec=cell.cell.soma[0])
        syn.e = 10.0
        syn.tau1 = 0.2
        syn.tau2 = 1.2

        return syn

    def somaCon(self, cell, syn,weight):

        netcon = nrn.NetCon(cell.cell.soma[0](0.5)._ref_v, syn,sec=cell.cell.soma[0])
        netcon.threshold = -10.0
        netcon.delay = 1.
        netcon.weight[0] = weight # nS

        return netcon

if __name__ == '__main__':

    print("main")

    #### Current steps ####

    template = RCTemplate()

    #input current steps
    stepcurrents0 = np.array([-150, -25, 20, 55, 60, 65, 70, -150]) * 1E-3
    stepcurrents1 = np.array([0, 0, 0, 0, 0, 0, 0, 20]) * 1E-3

    plt.figure()

    i = 0
    for stimamp0, stimamp1 in zip(stepcurrents0, stepcurrents1):

        neuron.h("forall delete_section()")
        nrn.celsius=36.0

        cell = template.return_cell()
        template.currentPulse(cell,stimamp0,stimamp1)

        plt.subplot(811+i, xlabel='Time [ms]', ylabel='mV')
        plt.plot(cell.tvec, cell.vmem[0, :])

        i+=1


    plt.show()
