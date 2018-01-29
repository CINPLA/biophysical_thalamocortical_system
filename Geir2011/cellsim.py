#!/usr/bin/env python

import os
import matplotlib.pyplot as plt
import numpy as np
from time import time
import LFPy
import neuron


class InterneuronTemplate(LFPy.TemplateCell):
    '''define Halnes et al 2011 interneuron model as a
    subclass of LFPy.TemplateCell'''
    def __init__(self,
                 
                #some LFPy params
                #morphology='ballnsticks.hoc',
                morphology='091008A2.hoc',
                #morphology='091008_A-2.hoc',
                passive=True,
                v_init = -63,
                extracellular=False,
                timeres_NEURON=0.1,
                timeres_python=0.1,
                tstopms=2000,
                tstartms=0,
                templatefile='LFPyCellTemplate.hoc',
                templatename='LFPyCellTemplate',
                templateargs=None,

                #model params
                e_pas = -67,
                Ra = 113,
                rm = 22000,
                cm = 1.1,
                gna =  0.09,
                nash = - 52.6,
                gkdr = 0.37,
                kdrsh = -51.2,
                gahp = 6.4e-5,
                gcat=1.17e-5,
                gcal=0.0009,
                ghbar=0.00011,
                catau = 50,
                gcanbar = 2e-8,
                hhdendfac = 0.1,
                ihdendfac = 1,
                ldendfac = 0.25,
                itinc = 2.39/60,
                actdends = 1,
                 **kwargs):
        
        #subclass LFPy.TemplateCell:
        LFPy.TemplateCell.__init__(self,
                morphology=morphology,
                passive = passive,
                v_init = v_init,
                e_pas = e_pas,
                Ra = Ra,
                rm = rm,
                cm = cm,
                extracellular=extracellular,
                timeres_NEURON=timeres_NEURON,
                timeres_python=timeres_python,
                tstopms=tstopms,
                tstartms=tstartms,
                templatefile=templatefile,
                templatename=templatename,
                templateargs=templateargs,
                **kwargs)
        
        #some factors
        iahpdendfac = hhdendfac
        #iahpdendfac = 0.1 #want to test unique hhdendfac
        
        icaninc = itinc
    
        neuron.h.celsius = 36.0
    
        neuron.h.distance()
        
        #Insert channels: (passive stuff already loaded in cell class)
        for sec in self.allseclist:
            sec.insert('iar')
            sec.ghbar_iar=ghbar*ihdendfac    #Ih-cation channel, slow, Zhu
            sec.insert('Cad')   # Calsium pool, Zhu et al.
            sec.insert('ical')  # L-type Ca-current, using pool in Cad
            sec.insert('it2')   # t-type Ca- current, using pool in Cad
            sec.insert('iahp')  # potassium current, slow, Ca-dependent, Zhu et al.
            sec.insert('hh2')
            sec.ena=50
            sec.ek=-90
            sec.insert('ican')  # CAN-channel from Zhu et al. 99a

        for sec in self.allseclist:
            sec.gkbar_hh2 = gkdr*hhdendfac
            sec.gnabar_hh2 = gkdr*hhdendfac*0.20
            sec.vtraubNa_hh2 = nash
            #sec.gkdrbar_hh2 = gkdr*hhdendfac
            sec.vtraubK_hh2 = kdrsh
            sec.pcabar_ical = gcal*ldendfac
            sec.gkbar_iahp = gahp*iahpdendfac
            sec.ghbar_iar = ghbar*ihdendfac
            sec.gcabar_it2 = gcat*(1 + itinc*neuron.h.distance(1))*actdends
            sec.gbar_ican = gcanbar*(1 + itinc*neuron.h.distance(1))*actdends
            
        for sec in self.somalist:
            sec.gnabar_hh2 = gna # gkdr*hhdendfac
            sec.vtraubNa_hh2 = nash 
            sec.gkbar_hh2 = gkdr
            sec.vtraubK_hh2 = kdrsh
            sec.gcabar_it2 = gcat
            sec.pcabar_ical = gcal
            sec.gkbar_iahp = gahp
            sec.ghbar_iar = ghbar
            sec.gbar_ican = gcanbar

        #for sec in self.allseclist:
        #    sec.gnabar_hh2 = gna*hhdendfac
        #    sec.vtraubNa_hh2 = nash
        #    #sec.gkdrbar_hh2 = gkdr*hhdendfac
        #    #sec.gkbar_hh2 = gkdr*hhdendfac
        #    sec.vtraubK_hh2 = kdrsh
        #    sec.pcabar_ical = gcal*ldendfac
        #    sec.gkbar_iahp = gahp*iahpdendfac
        #    sec.ghbar_iar = ghbar*ihdendfac
        #    sec.gcabar_it2 = gcat*(1 + itinc*neuron.h.distance(1))*actdends
        #    sec.gbar_ican = gcanbar*(1 + itinc*neuron.h.distance(1))*actdends
        #
        #for sec in self.somalist:
        #    sec.gnabar_hh2 = gna
        #    sec.vtraubNa_hh2 = nash 
        #    sec.gkbar_hh2 = gkdr
        #    sec.vtraubK_hh2 = kdrsh
        #    sec.gcabar_it2 = gcat
        #    sec.pcabar_ical = gcal
        #    sec.gkbar_iahp = gahp
        #    sec.ghbar_iar = ghbar
        #    sec.gbar_ican = gcanbar
    
        for sec in self.allseclist:    
            sec.taur_Cad = catau  # Calcium decay needs to know the the volume it enters


def cellsim(
        stimamp0=0.055,
        stimamp1=0.0,
        **kwargs):

    cell = InterneuronTemplate(**kwargs)
    
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

    
    return cell


################################################################################
#   MAIN
################################################################################

if __name__ == '__main__':
    
    neuron.h("forall delete_section()")
        
    ##original model parameters for P1
    #customCodeFunArgs = {
    #    'morphology' : 'ballnsticks.hoc',
    #    'e_pas' : -67,
    #    'Ra' : 113,
    #    'rm' : 22000,
    #    'cm' : 1.1,
    #    'gna' :  0.09,
    #    'nash' : - 52.6,
    #    'gkdr' : 0.37,
    #    'kdrsh' : -51.2,
    #    'gahp' : 6.4e-5,
    #    'gcat' : 1.17e-5,
    #    'gcal' : 0.0009,
    #    'ghbar' : 0.00011,
    #    'catau' :  50,
    #    'gcanbar' :  2e-8,
    #    'hhdendfac' :  0.1,
    #    'ihdendfac' : 1,
    #    'ldendfac' : 0.25,
    #    'itinc' : 2.39/60,
    #    'actdends' : 1,
    #}
    
    ##Geir's updated params
    #customCodeFunArgs = {
    #    'e_pas' : -67.5,
    #    'Ra' : 113,
    #    'rm' : 22000,
    #    'cm' : 1.1,
    #    'gna' :  0.09,
    #    'nash' : - 52.6,
    #    'gkdr' : 0.37,
    #    'kdrsh' : -51.2,
    #    'gahp' : 6.4e-5,
    #    'gcat' : 1.4e-4,
    #    'gcal' : 0.0009,
    #    'ghbar' : 0.00011,
    #    'catau' :  50,
    #    'gcanbar' :  6.8e-7,
    #    'hhdendfac' :  0.01,
    #    'ihdendfac' : 1,
    #    'ldendfac' : 0.25,
    #    'itinc' : 0,
    #    'actdends' : 1,
    #}

    # Geir's updated params
    customCodeFunArgs = {
        'morphology' : 'ballnsticks.hoc',
        'e_pas' : -67.5,
        'Ra' : 113,
        'rm' : 22000,
        'cm' : 1.1,
        'gna' :  0.1, #0.09,
        'nash' : - 52.6,
        'gkdr' : 0.37,
        'kdrsh' : -51.2,
        'gahp' : 6.4e-5,
        'gcat' : 1.4e-4,
        'gcal' : 0.0009,
        'ghbar' : 0.00011,
        'catau' :  50,
        'gcanbar' :  6.8e-7,
        'hhdendfac' :  0.1,
        'ihdendfac' : 1,
        'ldendfac' : 0.25,
        'itinc' : 0,
        'actdends' : 1,
    }

    
    #input current steps
    stepcurrents0 = np.array([-150, -25, 20, 55, 60, 65, 70, -150]) * 1E-3
    stepcurrents1 = np.array([0, 0, 0, 0, 0, 0, 0, 20]) * 1E-3
    #stepcurrents0 = np.array([65, 70, -150]) * 1E-3
    #stepcurrents1 = np.array([0, 0, 20]) * 1E-3
    
    fig0 = plt.figure()
    ax = fig0.add_subplot(111)
    
    i = 0
    for stimamp0, stimamp1 in zip(stepcurrents0, stepcurrents1):
        neuron.h("forall delete_section()")
        
        
        cell = cellsim(stimamp0=stimamp0, stimamp1=stimamp1, **customCodeFunArgs)
        
        ax.plot(cell.tvec, cell.somav-i, label='%.0f pA, %.0f pA' % (stimamp0*1E3, stimamp1*1E3))
        
        inds = (cell.tvec >= 100) & (cell.tvec <= 200)
        
        fig = plt.figure()
        imax = fig.add_subplot(111)
        im = imax.imshow(cell.vmem[:, inds])
        imax.set_title('%.0f pA, %.0f pA' % (stimamp0*1E3, stimamp1*1E3))
        imax.axis(imax.axis('tight'))
        fig.colorbar(im)
        
        i += 100 
    ax.axis(ax.axis('tight'))
    ax.set_xlabel('time (ms)')
    ax.set_ylabel(r'$V_\mathrm{soma}$ (mV)')
    ax.legend(loc='best')
    
    plt.show()
