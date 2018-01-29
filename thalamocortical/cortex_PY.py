#!/usr/bin/env python
# -*- coding: utf-8 -*-

## This class defines the properties and synapses of a cortical pyramidal cell.

from os.path import join
import numpy as np
import pylab as plt
from time import time
import neuron
import LFPy
import os

nrn = neuron.h

# Path to folder of neuron models
root_path = os.path.dirname(os.path.realpath(__file__))
root_path = root_path[0:root_path.find("thalamocortical")]

class CorticalPyramidalTemplate(object):

    def __init__(self):
        self.dummyP = 0.0

    def return_cell(self):

        self.model_path = join(root_path+'cortex_neurons')
        neuron.load_mechanisms(self.model_path)

        cell_parameters = {
            'morphology': join(self.model_path, 'soma.hoc'),
            'passive':False,
            'v_init' : -70,
            'extracellular':False,
            'nsegs_method': 'lambda_f',
            'lambda_f': 50,
            'timeres_NEURON':0.1,
            'timeres_python':0.1,
            'tstopms':2000,
            'tstartms':0,
            'templatefile': join(self.model_path, 'sPY_template'),
            'templatename':'sPY',
            'templateargs':None,
            'custom_fun': None,
            'delete_sections':False
        }

        cell = LFPy.TemplateCell(**cell_parameters)
        cell.set_rotation(x=-1.57, y=0.0, z=0.0)

        return cell


    def currentPulse(self,cell, stimamp0=0.055, stimamp1=0.0):

        PPparams0 = {
            'idx' : 0,
            'pptype' : 'IClamp',
            'delay' : 300,
            'dur' : 400,
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


    def TCConn(self, cell):
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

    def somaInhibition(self, cell):
        syn = nrn.Exp2Syn(0.5, sec=cell.cell.soma[0])
        syn.e = -60.0
        syn.tau1 = .45
        syn.tau2 = 5.0

        return syn



if __name__ == '__main__':

    print("main")

    #### Current steps ####

    template = CorticalPyramidalTemplate()

    plt.figure()

    neuron.h("forall delete_section()")
    nrn.celsius=36.0

    PY_neuron = template.return_cell()
    template.currentPulse(PY_neuron,0.75,0.0)

    plt.subplot(111, xlabel='Time [ms]', ylabel='mV')
    plt.plot(PY_neuron.tvec, PY_neuron.vmem[0, :])


    plt.show()
