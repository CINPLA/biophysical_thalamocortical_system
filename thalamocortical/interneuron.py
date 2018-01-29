#!/usr/bin/env python
# -*- coding: utf-8 -*-

## This class defines the properties and synapses of a dLGN interneuron.

from os.path import join
import numpy as np
#import pylab as plt
from time import time
import neuron
import LFPy
import os

import insertChannels
#reload(insertChannels)

nrn = neuron.h

# Path to folder of neuron models
root_path = os.path.dirname(os.path.realpath(__file__))
root_path = root_path[0:root_path.find("thalamocortical")]

class InterneuronTemplate(object):

    def __init__(self):

        self.PROXIMAL_SYN = dict(
            syntype='Exp2Syn',
            weight=0.0003,
            tau1=1.6,
            tau2=3.6,
            e=10.
        )

        self.DISTAL_SYN = dict(
            syntype='Exp2Syn',
            weight=0.0004,
            tau1=.3,
            tau2=2.,
            e=10.
        )



    def return_cell(self):

        self.model_path = join(root_path+'Geir2011')
        neuron.load_mechanisms(self.model_path)

        cell_parameters = {
            # 5 dendrites
            #'morphology': join(self.model_path, 'ballnsticks.hoc'),
            # 4 dendrites
            'morphology': join(self.model_path, 'ballnsticks2.hoc'),
            'passive':True,
            'v_init' : -63,
            'extracellular':False,
            'nsegs_method': 'lambda_f',
            'lambda_f': 50,
            'timeres_NEURON':0.1,
            'timeres_python':0.1,
            'tstopms':2000,
            'tstartms':0,
            'templatefile': join(self.model_path, 'LFPyCellTemplate.hoc'),
            'templatename':'LFPyCellTemplate',
            'templateargs':None,
            'custom_fun': [insertChannels.active_declarations],
            'custom_fun_args': [{'arg1': 'arg1'}],
            'delete_sections':False,

            # passive mechanisms
            'e_pas' : -67.5,
            'Ra' : 113,
            'rm' : 22000,
            'cm' : 1.1
        }

        cell = LFPy.TemplateCell(**cell_parameters)
        cell.set_rotation(x=-1.57, y=0.0, z=0.0)

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


    def createSynapse(self, cell, location, distal, spikes):

        if(distal==True):
            syn = LFPy.Synapse(cell, idx=location, **self.DISTAL_SYN)
        else:
            syn = LFPy.Synapse(cell, idx=location, **self.PROXIMAL_SYN)

        syn.set_spike_times(spikes)

        cell._loadspikes()

        return syn

    def somaCon(self, cell, syn):

        netcon = nrn.NetCon(cell.cell.soma(0.5)._ref_v, syn,sec=cell.cell.soma)
        netcon.threshold = -10.0
        netcon.delay = 1.
        netcon.weight[0] = 0.004

        return netcon


    def triadCon(self, cell, syn, syn_loc):

        i=0
        for sec in cell.allseclist:
            for seg in sec:
                if i == syn_loc:
                    sec_idx = sec
                    seg_idx = seg
                i+=1

        netcon = nrn.NetCon(seg_idx._ref_v, syn, sec=sec_idx)
        netcon.threshold = -30.0
        netcon.delay = 1.0
        netcon.weight[0] = 0.006

        return netcon

    def cortexCon(self, cell, location):

        i = 0
        for sec in cell.allseclist:
            for seg in sec:
                if (i==location):
                    syn = nrn.Exp2Syn(seg)
                    syn.tau1= 0.2
                    syn.tau2= 1.2
                    syn.e=10.
                i+=1

        return syn



if __name__ == '__main__':

    print("main")

    #### Current steps ####

    template = InterneuronTemplate()

    #input current steps
    stepcurrents0 = np.array([-150, -25, 20, 55, 60, 65, 70, -150]) * 1E-3
    stepcurrents1 = np.array([0, 0, 0, 0, 0, 0, 0, 20]) * 1E-3

    plt.figure()

    i = 0
    for stimamp0, stimamp1 in zip(stepcurrents0, stepcurrents1):

        neuron.h("forall delete_section()")
        nrn.celsius=36.0

        interneuron = template.return_cell()
        template.currentPulse(interneuron,stimamp0,stimamp1)

        plt.subplot(811+i, xlabel='Time [ms]', ylabel='mV')
        plt.plot(interneuron.tvec, interneuron.vmem[0, :])

        i+=1


    plt.show()
