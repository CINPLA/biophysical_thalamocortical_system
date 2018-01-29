#!/usr/bin/env python

import LFPy
import neuron
import numpy as np
import matplotlib.pyplot as plt

neuron.h("forall delete_section()")

cell = LFPy.TemplateCell(morphology='soma.hoc',
                         templatefile='TC_GH2.hoc',
                         templatename='sTC',
                         templateargs=None,
                         tstartms=0,
                         tstopms=5000,
                         v_init=-70,
                         passive=False,
                         extracellular=False,
                         nsegs_method=None,
                         )

#putative RGC spike trains
spiketrains = [
    np.random.rand(100).cumsum()*50+1000,
]

#container for synapses:
synapses = []
for i, spiketrain in enumerate(spiketrains):
    synapses.append(LFPy.Synapse(cell, idx=0, syntype='Exp2Syn',
                       weight=0.01, tau1=0.1, tau2=1, delay=1, ))
    synapses[i].set_spike_times(spiketrain)


##IClamps from original files:
#stim = LFPy.StimIntElectrode(cell, idx=0, pptype='IClamp',
#                             delay=2000, dur=1000, amp=0.045)
#stim2 = LFPy.StimIntElectrode(cell, idx=0, pptype='IClamp',
#                             delay=0, dur=5000, amp=0)

#set the temperature otherwise we get wrong responses
neuron.h.celsius=36.0

#run the sim:
cell.simulate()


plt.plot(cell.tvec, cell.somav)

plt.show()