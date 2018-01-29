#!/usr/bin/env python
# -*- coding: utf-8 -*-

## Insert active ion-channels in neuron models. This function loops over the
## sections of the morphology, defining which membrane mechanisms and corresponding
## densities and properties are present on the section.

import neuron
nrn = neuron.h

def biophys_active():
    # params
    gna =  0.1
    nash = - 52.6
    gkdr = 0.37
    kdrsh = -51.2
    gahp = 6.4e-5
    gcat = 1.8e-4
    gcal = 0.0009
    ghbar = 0.00011
    catau = 50
    gcanbar = 2e-8
    # Channel distribution (ratio: g_dend/g_soma)
    hhdendfac = 0.1
    ihdendfac = 1
    ldendfac = 0.25
    itinc = 0.0
    actdends = 1

    #some factors
    iahpdendfac = hhdendfac
    #iahpdendfac = 0.1 #want to test unique hhdendfac
    icaninc = itinc

    neuron.h.celsius = 36.0
    neuron.h.distance()

    for sec in neuron.h.allsec():   
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
        sec.gkbar_hh2 = gkdr*hhdendfac
        sec.gnabar_hh2 = gna*hhdendfac
        sec.vtraubNa_hh2 = nash
        #sec.gkdrbar_hh2 = gkdr*hhdendfac
        sec.vtraubK_hh2 = kdrsh
        sec.pcabar_ical = gcal*ldendfac
        sec.gkbar_iahp = gahp*iahpdendfac
        sec.ghbar_iar = ghbar*ihdendfac
        sec.gcabar_it2 = gcat*(1 + itinc*neuron.h.distance(1))*actdends
        sec.gbar_ican = gcanbar*(1 + itinc*neuron.h.distance(1))*actdends
        sec.taur_Cad = catau  # Calcium decay needs to know the the volume it enters

        if "soma" in sec.name():
            sec.gnabar_hh2 = gna # gkdr*hhdendfac
            sec.vtraubNa_hh2 = nash
            sec.gkbar_hh2 = gkdr
            sec.vtraubK_hh2 = kdrsh
            sec.gcabar_it2 = gcat
            sec.pcabar_ical = gcal
            sec.gkbar_iahp = gahp
            sec.ghbar_iar = ghbar
            sec.gbar_ican = gcanbar

    print("active ion-channels inserted")

def active_declarations(**kwargs):
    exec('biophys_active()')
