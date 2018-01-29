#!/usr/bin/env python

import os
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PolyCollection
import numpy as np
from time import time
import LFPy
import neuron
from cellsim import InterneuronTemplate



def simulate(cells,
             v_init=-60, tstartms=0, tstopms=100, dt=0.1,
             variable_dt=False, atol=0.001, step=False):
    '''
    Running the actual simulation in NEURON, simulations in NEURON
    is now interruptable.
    '''
    
    neuron.h.dt = dt
    
    cvode = neuron.h.CVode()
    
    #don't know if this is the way to do, but needed for variable dt method
    if variable_dt:
        cvode.active(1)
        cvode.atol(atol)
    else:
        cvode.active(0)
    
    #initialize state
    neuron.h.finitialize(v_init)
    
    #initialize current- and record
    if cvode.active():
        cvode.re_init()
    else:
        neuron.h.fcurrent()
    neuron.h.frecord_init()
    
    ##Starting simulation at tstart
    neuron.h.t = tstartms
    
    for cell in cells:
        cell._loadspikes()
    
    #print sim.time and realtime factor at intervals
    counter = 0.
    t0 = time()
    ti = neuron.h.t

    if step==False:
        neuron.run(tstopms)
        rtfactor = (neuron.h.t - ti)  * 1E-3 / (time() - t0)
        print 't = %.0f, realtime factor: %.3f' % (neuron.h.t, rtfactor)

    else:
        if tstopms > 1000:
            interval = 1 / dt * 100
        else:
            interval = 1 / dt * 10
    
        while neuron.h.t < tstopms:
            neuron.h.fadvance()
            counter += 1.
            if np.mod(counter, interval) == 0:
                rtfactor = (neuron.h.t - ti)  * 1E-3 / (time() - t0)
                print 't = %.0f, realtime factor: %.3f' % (neuron.h.t, rtfactor)
                t0 = time()
                ti = neuron.h.t
    
    if variable_dt:
        neuron.h.fadvance()

def _set_voltage_recorders(cell, tvec):
    '''
    Record membrane potentials for all segments
    '''
    cell.memvreclist = neuron.h.List()
    for sec in cell.allseclist:
        for seg in sec:
            memvrec = neuron.h.Vector()
            memvrec.record(seg._ref_v, tvec)
            cell.memvreclist.append(memvrec)

def get_idx_name(cell, idx=0):
    '''return NEURON name of segment idx,
    on the format cellname.sectionname(x)'''
    #create list of seg names:
    allsegnames = []
    for sec in cell.allseclist:
        for seg in sec:
            allsegnames.append('%s(%f)' % (sec.name(), seg.x))
    return np.array(allsegnames)[idx].tolist()


    
if __name__ == '__main__':
    
    neuron.h('forall delete_section()')
    
    dt = 2**-2
    tstartms = 0
    tstopms = 2000
    v_init = -70
    variable_dt=True
    
    simparams = {
        'timeres_NEURON' : dt,
        'timeres_python' : dt,
        'tstartms' : tstartms,
        'tstopms' : tstopms,
        'v_init' : v_init,
    }
    
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
    #    'hhdendfac' :  1, #0.01,
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
        'gna' :  0.09,
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
    
    
    synparams = {
        'syntype'  : 'Exp2Syn',
        'weight' : 0.001,
        'tau1' : 0.1,
        'tau2' : 0.75,
        'e' : 10.,
    }

    simparams.update(customCodeFunArgs)
    tvec = neuron.h.Vector(np.arange((tstopms-900) / dt+1)*dt + 900)
    
    #cell
    cell=InterneuronTemplate(**simparams)
    
    #rotate cell for visibility
    cell.set_rotation(x=np.pi/2)
    
    _set_voltage_recorders(cell, tvec)
    
    #somatic synapse, chosing a very small weight to keep things linearish
    syn = LFPy.Synapse(cell, idx=0, **synparams)
    syn.set_spike_times(np.array([920]))

    #PPparams0 = {
    #    'idx' : 0,
    #    'pptype' : 'IClamp',
    #    'delay' : 920,
    #    'dur' : 1000,
    #    'amp' : 0.070,
    #}
    #stim0 = LFPy.StimIntElectrode(cell, **PPparams0)


    
    #neuron.h.celsius=36.0    
    simulate([cell], v_init=v_init, tstartms=tstartms, tstopms=tstopms, dt=dt,
        step=True, variable_dt=variable_dt)
    
    #collect
    cell._collect_vmem()
    
    
    #identify some idx with some EPSP amplitude we want:
    EPSPs = cell.vmem - cell.vmem[:, np.array(tvec)==920-dt]
    EPSPamp = EPSPs.max(axis=1)
    EPSPamp /= EPSPamp[0] #norm by EPSPamp[0] 
    
    #bins for normed EPSP amplitudes
    normbins = zip(np.arange(0, 1, 0.1), np.arange(0, 1, 0.1)+0.1)
    #normbins = zip(np.arange(0, 1, 0.05), np.arange(0, 1, 0.05)+0.05)
    
    #create a nested list of idx&name pairs with EPSP amplitudes for each bin
    zip_idx_names = []
    for i, somebin in enumerate(normbins):
        idxs = np.where((EPSPamp >= somebin[0]) & (EPSPamp < somebin[1]))[0]
        names = get_idx_name(cell, idxs)
        #filter out neighbouring idxs that are on the same section
        lastname = None
        zip_idx_names.append([])
        for idx, name in zip(idxs, names):
            if name.split('(')[0] != lastname:
                zip_idx_names[i].append((idx, name))
            
            lastname = name.split('(')[0]
                 


    #plt.plot(tvec, cell.vmem[0, ])
    #plt.plot(tvec, cell.vmem[410, ])
    #plt.axis('tight')
    #plt.show()
    #raise Exception    

    

    ############################################################################
    #plot norm EPSP amplitudes 
    fig = plt.figure()
    ax = fig.add_subplot(111)
    zips = []
    for x, z in cell.get_idx_polygons():
        zips.append(zip(x, z))
    polycol = PolyCollection(zips,
                             edgecolors='none',
                             facecolors='gray')            
    ax.add_collection(polycol)
    im = ax.scatter(cell.xmid, cell.zmid, s=40, c=EPSPamp, edgecolors='none')
    ax.axis('equal')
    cbar=fig.colorbar(im)
    cbar.set_label('norm EPSP ampl')
    ax.set_title('rel. EPSP ampl.')
    fig.savefig('EPSPv1.pdf', dpi=150)
    
    #plot each EPSP    
    plt.figure()
    for trace in EPSPs:
        plt.plot(tvec, trace)
    plt.axis('tight')
    plt.title('EPSP amplitudes (mV)')
    plt.savefig('EPSPv2.pdf', dpi=150)


    #raise Exception

    ############################################################################
    #run a series of simulations testing all these synaptic locations
    for i, idx_names in enumerate(zip_idx_names):
        if len(idx_names) != 0:

            fig = plt.figure(figsize=(16, 8))
            ax = fig.add_subplot(121)

            
            zips = []
            for x, z in cell.get_idx_polygons():
                zips.append(zip(x, z))
            polycol = PolyCollection(zips,
                                     edgecolors='none',
                                     facecolors='gray')            
            ax = fig.add_subplot(121)
            ax.add_collection(polycol)

            
            #plot syn locations
            for idx, name in idx_names:
                plt.scatter(cell.xmid[idx], cell.zmid[idx], s=40, c='r',
                            edgecolors='none')
        
            ax.axis(ax.axis('equal'))
            ax.set_title('norm EPSP ampl.: [%.2f, %.2f>' % (normbins[i]))


            ax2 = fig.add_subplot(222)
            ax3 = fig.add_subplot(224)

            for idx, name in idx_names:
                neuron.h('forall delete_section()')
                
                cell=InterneuronTemplate(**simparams)
                _set_voltage_recorders(cell, tvec)

                #rotate cell for visibility
                cell.set_rotation(x=np.pi/2)
                
                syn = LFPy.Synapse(cell, idx=idx, **synparams)
                syn.set_spike_times(np.array([920]))
                
                simulate([cell], v_init=v_init, tstartms=tstartms,
                    tstopms=tstopms, dt=dt, step=True, variable_dt=variable_dt)
                
                #collect
                cell._collect_vmem()
                
            
                ax2.plot(tvec, cell.vmem[0, ]-cell.vmem[0, np.array(tvec)==920-dt],
                         label='idx %i' % idx)
                ax3.plot(tvec, cell.vmem[idx, ],
                         label='idx %i' % idx)
                
                
            ax2.set_title('somatic EPSP: [%.2f, %.2f>' % (normbins[i]))
            leg = ax2.legend(loc='best')
            for t in leg.get_texts():
                t.set_fontsize('x-small')    # the legend text fontsize
            for idx, name in idx_names:
                ax2.plot(tvec, EPSPs[idx, ], 'k')
            ax2.axis(ax2.axis('tight'))

            ax3.set_title('idx EPSP: [%.2f, %.2f>' % (normbins[i]))
            leg = ax3.legend(loc='best')
            for t in leg.get_texts():
                t.set_fontsize('x-small')    # the legend text fontsize
            
            fig.savefig('idxpos%.2i.pdf' % i, dpi=150)

            
    plt.show()



    


    