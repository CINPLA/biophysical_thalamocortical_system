#!/usr/bin/env python
# -*- coding: utf-8 -*-

## Simulation of the thalamocortical network. Retinal cells' spikes are loaded
## from file. The arrangement of the cortical feedback (spatial extent and
## synaptic weights) can be modified in Parameters.

import os
import sys
import time
import numpy as np
import neuron
import LFPy
from mpi4py import MPI

# Initialize the MPI environment
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

# Import neuron models
import ganglionCell
#reload(ganglionCell)

import interneuron
#reload(interneuron)

import relayCell
#reload(relayCell)

import cortex_PY
#reload(cortex_PY)

# Import auxiliary libraries
import simulation
#reload(simulation)

import topology
#reload(topology)

#import dataPlot
#reload(dataPlot)

# Path to folder 'thalamocortical'
root_path = os.path.dirname(os.path.realpath(__file__))+"/"

#############################
#### Start of parameters ####
#############################

# Number of neurons per row (all layers except INs)
N = 10.0
# Ratio of interneurons (25 %)
INratio = 0.25
# Visual angle
visSize = 10.0

# Mask of interneurons (number of retinal cells to connect)
IN_RC_mask = 4

# Relay cell -> Pyramidal cell (horizontal and vertical orientations)
# Receptive-field center
RC_PY_mask_h = [1,3]
RC_PY_mask_v = [3,1]
# Receptive-field surround
RC_PY_s_mask_h = [1,3]
RC_PY_s_mask_v = [3,1]

# Feedback masks
# Pyramidal cell -> Relay cell
PY_RC_mask = [2,2]
# Pyramidal cell -> interneuron (number of interneurons to connect)
PY_IN_mask = 4

# Type of feedback (0 = phase-reversed, 1 = phase-matched, 2 = No feedback)
feedback_type = 2

# Stimulus setup
# Stimulus type (0 = flashing circular spot, 1 = patch grating, 2 = receptive field)
stimulustype = 0

# Type of spot (used only for the receptive field). RF_1 = white spot, RF_2 =
# black spot
spot_type_tha = "RF_1" # thalamocortical label
spot_type_ret = "RF_1" # retina label

# Selected spot/patch diameter to save intracellular potentials
stim_to_plot = 2.0

# Cell ID to save intracellular potentials
cell_number = 55 # all layers except INs
cell_number_IN = 12

# Simulation parameters
tsim = 1000.0
#tsim = 300.0
dt = 0.2
binsize = 5.0
numbertrials = 5.0

# Feedback parameter combinations
p1 = [0.006/4.0] # PY -> RC
p2 = [0.0012/4.0] # PY -> IN

#p1 = [0.0, 0.001/1.0, 0.002/1.0, 0.006/1.0] # PY -> RC
#p2 = [0.0, 0.0005/1.0, 0.001/1.0, 0.006/1.0] # PY -> IN

#############################
##### End of parameters #####
#############################

# Create stimulus
if stimulustype <= 1:
    # Range of spot/patch diameters (deg)
    stimulus = np.arange(0.0,10.2,0.2)

else:
    # Receptive field
    Npoints = 8.0 # Number of flashing spots per row
    raw_stimulus = np.arange(0.0,Npoints*Npoints,1.0)
    stimulus = []
    # To speed computations, select a center square
    row_top_limit = 0.0 # (between 0 and N)
    row_bottom_limit = 4.0 # (between 0 and N)
    col_left_limit = 0.0 # (between 0 and N)
    col_right_limit = 4.0 # (between 0 and N)

    for pos in raw_stimulus:
        row = int(pos/Npoints)*(N/Npoints)
        col = np.remainder(pos,Npoints)*(N/Npoints)

        if(row >= row_top_limit and row <= row_bottom_limit and col >=\
        col_left_limit and col <= col_right_limit):
            stimulus.append(pos)

# Folder to save results
if stimulustype == 0:
    type = "disk/comb"
    type_ret = "disk"
elif stimulustype == 1:
    type = "patch_grating/0/comb"
    type_ret = "patch_grating/0"
elif stimulustype == 2:
    type = spot_type_tha+"/comb"
    type_ret = spot_type_ret

# Create folders
if rank == 0:
    os.system("mkdir "+root_path+"results/thalamocortical")
    if stimulustype == 0:
        os.system("mkdir "+root_path+"results/thalamocortical/disk")
    elif stimulustype == 1:
        os.system("mkdir "+root_path+"results/thalamocortical/patch_grating")
        os.system("mkdir "+root_path+"results/thalamocortical/patch_grating/0")
    elif stimulustype == 2:
        os.system("mkdir "+root_path+"results/thalamocortical/"+spot_type_tha)

# Arrays with all combinations of feedback weights
p1_ext = []
p2_ext = []
for pp1 in p1:
    for pp2 in p2:
        p1_ext.append(pp1)
        p2_ext.append(pp2)

# Synaptic parameters of interneuron (4 dendrites)
syn_base_ids = np.array([1, 24, 47, 70])
# GC -> IN (proximal interneuron dendrite)
syn_comps_prox = syn_base_ids + 3
# GC -> IN (distal interneuron dendrite)
syn_comps_dist = syn_base_ids + 20
# Feedback synapse from pyramidal cells
syn_comps_int = syn_base_ids + 18


#############################
######## MPI function #######
#############################

def worker(stimulus,stimulus_type):

    # Neuron imports hoc and does a  nrn = hoc.HocObject()
    nrn = neuron.h
    # To avoid Neuron displaying messages
    copystdout = sys.stdout
    # Print info of process
    print("Rank = %s, Stimulus = %s, Experiment = %s" % (rank, stimulus,stimulus_type))

    comb = 0
    for stim in stimulus:
        # Reset PSTHs
        PST_IN_ON = np.zeros((int(N*N*INratio),int(tsim/binsize)))
        PST_IN_OFF = np.zeros((int(N*N*INratio),int(tsim/binsize)))
        PST_RC_ON = np.zeros((int(N*N),int(tsim/binsize)))
        PST_RC_OFF = np.zeros((int(N*N),int(tsim/binsize)))
        PST_CXPY_h_ON = np.zeros((int(N*N),int(tsim/binsize)))
        PST_CXPY_v_ON = np.zeros((int(N*N),int(tsim/binsize)))
        PST_CXPY_h_OFF = np.zeros((int(N*N),int(tsim/binsize)))
        PST_CXPY_v_OFF = np.zeros((int(N*N),int(tsim/binsize)))

        for trial in np.arange(0,numbertrials):
            # Remove all sections and set temperature
            nrn("forall delete_section()")
            nrn.celsius=36.0

            # To avoid Neuron displaying messages
            f = open('/dev/null', 'w')
            sys.stdout = f

            # Simulation and topology modules
            sim = simulation.Simulation()
            tp = topology.Topology()

            # Reset spike counters
            spikes_IN_ON = []
            spikes_IN_OFF = []
            spikes_RC_ON = []
            spikes_RC_OFF = []
            spikes_CXPY_h_ON = []
            spikes_CXPY_v_ON = []
            spikes_CXPY_h_OFF = []
            spikes_CXPY_v_OFF = []

            # to store Neuron cells
            NEURON_cells_to_sim = []

            # to remember positions of IN's dendrites
            IN_dend_dict = np.zeros([N*N,2],dtype=int)

            #################################
            ### Creation of neuron models ###
            #################################

            #### Retina ####
            retina = ganglionCell.Retina()
            spikes_ON = retina.loadSpikes(stim,trial,N,type_ret+"/ON")
            spikes_OFF = retina.loadSpikes(stim,trial,N,type_ret+"/OFF")

            ### Interneuron ###
            INs_ON = []
            INs_OFF = []
            template1 = interneuron.InterneuronTemplate()

            n = 0
            for x in np.arange(0,N*np.sqrt(INratio)):
                for y in np.arange(0,N*np.sqrt(INratio)):
                    INs_ON.append(template1.return_cell())
                    NEURON_cells_to_sim.append(INs_ON[n])
                    INs_OFF.append(template1.return_cell())
                    NEURON_cells_to_sim.append(INs_OFF[n])

                    INs_ON[n].tstopms = tsim
                    INs_ON[n].tstartms = 0.0
                    INs_ON[n].timeres_NEURON = dt
                    INs_ON[n].timeres_python = dt
                    INs_OFF[n].tstopms = tsim
                    INs_OFF[n].tstartms = 0.0
                    INs_OFF[n].timeres_NEURON = dt
                    INs_OFF[n].timeres_python = dt

                    # spike counter
                    exec( 'apc_IN_ON%s = nrn.APCount(INs_ON[n].cell.soma(0.5))' % n)
                    exec( 'apc_IN_ON%s.thresh = -10.0' % n)
                    exec( 'apc_IN_count_ON%s = nrn.Vector(1)' % n)
                    exec( 'apc_IN_ON%s.record(apc_IN_count_ON%s)' % (n,n))
                    exec( 'spikes_IN_ON.append(apc_IN_count_ON%s)' % n)

                    exec( 'apc_IN_OFF%s = nrn.APCount(INs_OFF[n].cell.soma(0.5))' % n)
                    exec( 'apc_IN_OFF%s.thresh = -10.0' % n)
                    exec( 'apc_IN_count_OFF%s = nrn.Vector(1)' % n)
                    exec( 'apc_IN_OFF%s.record(apc_IN_count_OFF%s)' % (n,n))
                    exec( 'spikes_IN_OFF.append(apc_IN_count_OFF%s)' % n)

                    syns = tp.fitSquare(N,IN_RC_mask,int(x/np.sqrt(INratio))*N + int(y/np.sqrt(INratio)))

                    # Retina inputs
                    count = 0
                    for sn in syns:
                        template1.createSynapse(INs_ON[n], syn_comps_dist[count], True, np.array(spikes_ON[int(sn)]))
                        template1.createSynapse(INs_ON[n], syn_comps_prox[count], False, np.array(spikes_ON[int(sn)]))
                        template1.createSynapse(INs_OFF[n], syn_comps_dist[count], True, np.array(spikes_OFF[int(sn)]))
                        template1.createSynapse(INs_OFF[n], syn_comps_prox[count], False, np.array(spikes_OFF[int(sn)]))
                        # Keep position of dendrite and corresponding IN
                        IN_dend_dict[sn,:] = [n,count]
                        count+=1

                    n+=1

            #### Relay cell ####
            RCs_ON = []
            RCs_OFF = []
            template2 = relayCell.RCTemplate()

            n = 0
            for x in np.arange(0,N):
                for y in np.arange(0,N):
                    RCs_ON.append(template2.return_cell())
                    NEURON_cells_to_sim.append(RCs_ON[n])
                    RCs_OFF.append(template2.return_cell())
                    NEURON_cells_to_sim.append(RCs_OFF[n])

                    RCs_ON[n].tstopms = tsim
                    RCs_ON[n].tstartms = 0.0
                    RCs_ON[n].timeres_NEURON = dt
                    RCs_ON[n].timeres_python = dt
                    RCs_OFF[n].tstopms = tsim
                    RCs_OFF[n].tstartms = 0.0
                    RCs_OFF[n].timeres_NEURON = dt
                    RCs_OFF[n].timeres_python = dt

                    # spike counter
                    exec( 'apc_RC_ON%s = nrn.APCount(RCs_ON[n].cell.soma[0](0.5))' % n)
                    exec( 'apc_RC_ON%s.thresh = -10.0' % n)
                    exec( 'apc_RC_count_ON%s = nrn.Vector(1)' % n)
                    exec( 'apc_RC_ON%s.record(apc_RC_count_ON%s)' % (n,n))
                    exec( 'spikes_RC_ON.append(apc_RC_count_ON%s)' % n)

                    exec( 'apc_RC_OFF%s = nrn.APCount(RCs_OFF[n].cell.soma[0](0.5))' % n)
                    exec( 'apc_RC_OFF%s.thresh = -10.0' % n)
                    exec( 'apc_RC_count_OFF%s = nrn.Vector(1)' % n)
                    exec( 'apc_RC_OFF%s.record(apc_RC_count_OFF%s)' % (n,n))
                    exec( 'spikes_RC_OFF.append(apc_RC_count_OFF%s)' % n)

                    # Retina inputs
                    template2.createSynapse(RCs_ON[n], 0, False, np.array(spikes_ON[n]), 1.0)
                    template2.createSynapse(RCs_OFF[n], 0, False, np.array(spikes_OFF[n]), 1.0)

                    n+=1

            #### Cortical pyramidal cell ####
            PYs_h_ON = []
            PYs_v_ON = []
            PYs_h_OFF = []
            PYs_v_OFF = []
            template3 = cortex_PY.CorticalPyramidalTemplate()

            n = 0
            for x in np.arange(0,N):
                for y in np.arange(0,N):
                    PYs_h_ON.append(template3.return_cell())
                    PYs_v_ON.append(template3.return_cell())
                    NEURON_cells_to_sim.append(PYs_h_ON[n])
                    NEURON_cells_to_sim.append(PYs_v_ON[n])
                    PYs_h_OFF.append(template3.return_cell())
                    PYs_v_OFF.append(template3.return_cell())
                    NEURON_cells_to_sim.append(PYs_h_OFF[n])
                    NEURON_cells_to_sim.append(PYs_v_OFF[n])

                    PYs_h_ON[n].tstopms = tsim
                    PYs_h_ON[n].tstartms = 0.0
                    PYs_h_ON[n].timeres_NEURON = dt
                    PYs_h_ON[n].timeres_python = dt

                    PYs_v_ON[n].tstopms = tsim
                    PYs_v_ON[n].tstartms = 0.0
                    PYs_v_ON[n].timeres_NEURON = dt
                    PYs_v_ON[n].timeres_python = dt

                    PYs_h_OFF[n].tstopms = tsim
                    PYs_h_OFF[n].tstartms = 0.0
                    PYs_h_OFF[n].timeres_NEURON = dt
                    PYs_h_OFF[n].timeres_python = dt

                    PYs_v_OFF[n].tstopms = tsim
                    PYs_v_OFF[n].tstartms = 0.0
                    PYs_v_OFF[n].timeres_NEURON = dt
                    PYs_v_OFF[n].timeres_python = dt

                    # spike counter
                    exec( 'apc_PYs_h_ON%s = nrn.APCount(PYs_h_ON[n].cell.soma[0](0.5))' % n)
                    exec( 'apc_PYs_h_ON%s.thresh = -10.0' % n)
                    exec( 'apc_PYs_h_count_ON%s = nrn.Vector(1)' % n)
                    exec( 'apc_PYs_h_ON%s.record(apc_PYs_h_count_ON%s)' % (n,n))
                    exec( 'spikes_CXPY_h_ON.append(apc_PYs_h_count_ON%s)' % n)

                    exec( 'apc_PYs_v_ON%s = nrn.APCount(PYs_v_ON[n].cell.soma[0](0.5))' % n)
                    exec( 'apc_PYs_v_ON%s.thresh = -10.0' % n)
                    exec( 'apc_PYs_v_count_ON%s = nrn.Vector(1)' % n)
                    exec( 'apc_PYs_v_ON%s.record(apc_PYs_v_count_ON%s)' % (n,n))
                    exec( 'spikes_CXPY_v_ON.append(apc_PYs_v_count_ON%s)' % n)

                    exec( 'apc_PYs_h_OFF%s = nrn.APCount(PYs_h_OFF[n].cell.soma[0](0.5))' % n)
                    exec( 'apc_PYs_h_OFF%s.thresh = -10.0' % n)
                    exec( 'apc_PYs_h_count_OFF%s = nrn.Vector(1)' % n)
                    exec( 'apc_PYs_h_OFF%s.record(apc_PYs_h_count_OFF%s)' % (n,n))
                    exec( 'spikes_CXPY_h_OFF.append(apc_PYs_h_count_OFF%s)' % n)

                    exec( 'apc_PYs_v_OFF%s = nrn.APCount(PYs_v_OFF[n].cell.soma[0](0.5))' % n)
                    exec( 'apc_PYs_v_OFF%s.thresh = -10.0' % n)
                    exec( 'apc_PYs_v_count_OFF%s = nrn.Vector(1)' % n)
                    exec( 'apc_PYs_v_OFF%s.record(apc_PYs_v_count_OFF%s)' % (n,n))
                    exec( 'spikes_CXPY_v_OFF.append(apc_PYs_v_count_OFF%s)' % n)

                    n+=1

            ############################
            ### Synaptic connections ###
            ############################

            general_counter = 0

            #### Interneuron -> RC connection
            n = 0
            for x in np.arange(0,N*np.sqrt(INratio)):
                for y in np.arange(0,N*np.sqrt(INratio)):

                    syns = tp.fitSquare(N,IN_RC_mask,int(x/np.sqrt(INratio))*N + int(y/np.sqrt(INratio)))
#                    print(syns)
#                    tp.showConn(N,syns)
                    count = 0

                    for sn in syns:
                        # triadic synapse
                        exec( 'syn_ON%s = template2.triadSynapse(RCs_ON[int(sn)])' % str(general_counter))
                        exec( 'netcon_ON%s = template1.triadCon(cell=INs_ON[n], syn=syn_ON%s, '\
                        'syn_loc=syn_comps_dist[count])' % (str(general_counter),str(general_counter)))

                        # soma connection
                        exec( 'syn_ON%s = template2.somaInhibition(RCs_ON[int(sn)])' % str(general_counter+1))
                        exec( 'netcon_ON%s = template1.somaCon(cell=INs_ON[n], '\
                        'syn=syn_ON%s)' % (str(general_counter+1),str(general_counter+1)))

                        # triadic synapse
                        exec( 'syn_OFF%s = template2.triadSynapse(RCs_OFF[int(sn)])' % str(general_counter))
                        exec( 'netcon_OFF%s = template1.triadCon(cell=INs_OFF[n], syn=syn_OFF%s, '\
                        'syn_loc=syn_comps_dist[count])' % (str(general_counter),str(general_counter)))

                        # soma connection
                        exec( 'syn_OFF%s = template2.somaInhibition(RCs_OFF[int(sn)])' % str(general_counter+1))
                        exec( 'netcon_OFF%s = template1.somaCon(cell=INs_OFF[n], '\
                        'syn=syn_OFF%s)' % (str(general_counter+1),str(general_counter+1)))

                        count+=1
                        general_counter+=2

                    n+=1

            #### RC connection -> PY cells (Receptive-field center)
            n = 0
            for x in np.arange(0,N):
                for y in np.arange(0,N):

                    synsh = tp.rectMask(N,RC_PY_mask_h,n)
#                    print(synsh)
#                    tp.showConn(N,synsh)

                    count = 0

                    for sn in synsh:
                        exec( 'syn_ON%s = template3.TCConn(PYs_h_ON[int(n)])' % str(general_counter))
                        exec( 'netcon_ON%s = template2.somaCon(cell=RCs_ON[int(sn)], '\
                        'syn=syn_ON%s,weight=0.05)' % (str(general_counter),str(general_counter)))
                        exec( 'syn_OFF%s = template3.TCConn(PYs_h_OFF[int(n)])' % str(general_counter))
                        exec( 'netcon_OFF%s = template2.somaCon(cell=RCs_OFF[int(sn)], '\
                        'syn=syn_OFF%s,weight=0.05)' % (str(general_counter),str(general_counter)))

                        count+=1
                        general_counter+=1

                    synsv = tp.rectMask(N,RC_PY_mask_v,n)
#                    print(synsv)
#                    tp.showConn(N,synsv)

                    count = 0

                    for sn in synsv:
                        exec( 'syn_ON%s = template3.TCConn(PYs_v_ON[int(n)])' % str(general_counter))
                        exec( 'netcon_ON%s = template2.somaCon(cell=RCs_ON[int(sn)], '\
                        'syn=syn_ON%s,weight=0.05)' % (str(general_counter),str(general_counter)))
                        exec( 'syn_OFF%s = template3.TCConn(PYs_v_OFF[int(n)])' % str(general_counter))
                        exec( 'netcon_OFF%s = template2.somaCon(cell=RCs_OFF[int(sn)], '\
                        'syn=syn_OFF%s,weight=0.05)' % (str(general_counter),str(general_counter)))

                        count+=1
                        general_counter+=1

                    n+=1

            #### RC connection -> PY cells (Receptive-field surround)
            n = 0
            for x in np.arange(0,N):
                for y in np.arange(0,N):

                    synsh = tp.rectMask(N,RC_PY_s_mask_h,n+N)
#                    print(synsh)
#                    tp.showConn(N,synsh)

                    count = 0

                    for sn in synsh:
                        exec( 'syn_ON%s = template3.TCConn(PYs_h_OFF[int(n)])' % str(general_counter))
                        exec( 'netcon_ON%s = template2.somaCon(cell=RCs_ON[int(sn)], '\
                        'syn=syn_ON%s,weight=0.02)' % (str(general_counter),str(general_counter)))
                        exec( 'syn_OFF%s = template3.TCConn(PYs_h_ON[int(n)])' % str(general_counter))
                        exec( 'netcon_OFF%s = template2.somaCon(cell=RCs_OFF[int(sn)], '\
                        'syn=syn_OFF%s,weight=0.02)' % (str(general_counter),str(general_counter)))

                        count+=1
                        general_counter+=1

                    synsv = tp.rectMask(N,RC_PY_s_mask_v,n+1)
#                    print(synsv)
#                    tp.showConn(N,synsv)

                    count = 0

                    for sn in synsv:
                        exec( 'syn_ON%s = template3.TCConn(PYs_v_OFF[int(n)])' % str(general_counter))
                        exec( 'netcon_ON%s = template2.somaCon(cell=RCs_ON[int(sn)], '\
                        'syn=syn_ON%s,weight=0.02)' % (str(general_counter),str(general_counter)))
                        exec( 'syn_OFF%s = template3.TCConn(PYs_v_ON[int(n)])' % str(general_counter))
                        exec( 'netcon_OFF%s = template2.somaCon(cell=RCs_OFF[int(sn)], '\
                        'syn=syn_OFF%s,weight=0.02)' % (str(general_counter),str(general_counter)))

                        count+=1
                        general_counter+=1

                    n+=1

            #### Feedback: PY cells -> RC cells

            if (feedback_type < 2):

                n = 0
                for x in np.arange(0,N):
                    for y in np.arange(0,N):

                        synRC = tp.rectMask(N,PY_RC_mask,n)
    #                    print(synRC)
    #                    tp.showConn(N,synRC)

                        count = 0

                        for sn in synRC:

                            ## Phase-reversed feedback
                            if (feedback_type == 0):

                                exec( 'syn_ON%s = template2.somaExcitation(RCs_ON[int(sn)])' % str(general_counter))
                                exec( 'netcon_ON%s = template3.somaCon(cell=PYs_h_OFF[int(n)], '\
                                'syn=syn_ON%s,weight=p1_ext[stimulus_type[comb]])'
                                 % (str(general_counter),str(general_counter)))

                                exec( 'syn_ON%s = template2.somaExcitation(RCs_ON[int(sn)])' % str(general_counter+1))
                                exec( 'netcon_ON%s = template3.somaCon(cell=PYs_v_OFF[int(n)], '\
                                'syn=syn_ON%s,weight=p1_ext[stimulus_type[comb]])'
                                 % (str(general_counter+1),str(general_counter+1)))

                                exec( 'syn_OFF%s = template2.somaExcitation(RCs_OFF[int(sn)])' % str(general_counter))
                                exec( 'netcon_OFF%s = template3.somaCon(cell=PYs_h_ON[int(n)], '
                                'syn=syn_OFF%s,weight=p1_ext[stimulus_type[comb]])'
                                 % (str(general_counter),str(general_counter)))

                                exec( 'syn_OFF%s = template2.somaExcitation(RCs_OFF[int(sn)])' % str(general_counter+1))
                                exec( 'netcon_OFF%s = template3.somaCon(cell=PYs_v_ON[int(n)], '\
                                'syn=syn_OFF%s,weight=p1_ext[stimulus_type[comb]])'
                                % (str(general_counter+1),str(general_counter+1)))

                            ## Phase-matched feedback
                            else:
                                exec( 'syn_ON%s = template2.somaExcitation(RCs_ON[int(sn)])' % str(general_counter))
                                exec( 'netcon_ON%s = template3.somaCon(cell=PYs_h_ON[int(n)], '\
                                'syn=syn_ON%s,weight=p1_ext[stimulus_type[comb]])'
                                 % (str(general_counter),str(general_counter)))

                                exec( 'syn_ON%s = template2.somaExcitation(RCs_ON[int(sn)])' % str(general_counter+1))
                                exec( 'netcon_ON%s = template3.somaCon(cell=PYs_v_ON[int(n)], '\
                                'syn=syn_ON%s,weight=p1_ext[stimulus_type[comb]])'
                                 % (str(general_counter+1),str(general_counter+1)))

                                exec( 'syn_OFF%s = template2.somaExcitation(RCs_OFF[int(sn)])' % str(general_counter))
                                exec( 'netcon_OFF%s = template3.somaCon(cell=PYs_h_OFF[int(n)], '\
                                'syn=syn_OFF%s,weight=p1_ext[stimulus_type[comb]])'
                                 % (str(general_counter),str(general_counter)))

                                exec( 'syn_OFF%s = template2.somaExcitation(RCs_OFF[int(sn)])' % str(general_counter+1))
                                exec( 'netcon_OFF%s = template3.somaCon(cell=PYs_v_OFF[int(n)], '\
                                'syn=syn_OFF%s,weight=p1_ext[stimulus_type[comb]])'
                                 % (str(general_counter+1),str(general_counter+1)))

                            count+=1
                            general_counter+=2

                        n+=1

            #### Feedback: PY cells -> INs

            if (feedback_type < 2):

                n = 0
                for x in np.arange(0,N):
                    for y in np.arange(0,N):

                        synRC = tp.rectMask(N,PY_RC_mask,n)
    #                    print(synRC)
    #                    tp.showConn(N,synRC)

                        for sn in synRC:
                            exec( 'synd_ON%s = template1.cortexCon(INs_ON[IN_dend_dict[sn,0]], '\
                            'syn_comps_int[IN_dend_dict[sn,1]])' % str(general_counter))
                            exec( 'netcon_ON%s = template3.somaCon(cell=PYs_h_ON[int(n)], '\
                            'syn=synd_ON%s,weight=p2_ext[stimulus_type[comb]])' % (str(general_counter),str(general_counter)))

                            exec( 'synd_ON%s = template1.cortexCon(INs_ON[IN_dend_dict[sn,0]], '\
                            'syn_comps_int[IN_dend_dict[sn,1]])' % str(general_counter+1))
                            exec( 'netcon_ON%s = template3.somaCon(cell=PYs_v_ON[int(n)], syn=synd_ON%s, '\
                            'weight=p2_ext[stimulus_type[comb]])' % (str(general_counter+2),str(general_counter+1)))

                            exec( 'synd_OFF%s = template1.cortexCon(INs_OFF[IN_dend_dict[sn,0]], '\
                            'syn_comps_int[IN_dend_dict[sn,1]])' % str(general_counter))
                            exec( 'netcon_OFF%s = template3.somaCon(cell=PYs_h_OFF[int(n)], syn=synd_OFF%s, '\
                            'weight=p2_ext[stimulus_type[comb]])' % (str(general_counter),str(general_counter)))

                            exec( 'synd_OFF%s = template1.cortexCon(INs_OFF[IN_dend_dict[sn,0]], '\
                            'syn_comps_int[IN_dend_dict[sn,1]])' % str(general_counter+1))
                            exec( 'netcon_OFF%s = template3.somaCon(cell=PYs_v_OFF[int(n)], syn=synd_OFF%s, '\
                            'weight=p2_ext[stimulus_type[comb]])' % (str(general_counter+2),str(general_counter+1)))

                            general_counter+=4

                        n+=1

            #### Simulation
            sim.simulateCells(NEURON_cells_to_sim)
            sys.stdout = copystdout

            # update PSTHs
            for n in np.arange(0,len(INs_ON)):
                h, e = np.histogram(np.array(spikes_IN_ON[n]), bins=np.arange(0., tsim+binsize, binsize))
                PST_IN_ON[n,:] += h * (1000. / binsize)
            for n in np.arange(0,len(INs_OFF)):
                h, e = np.histogram(np.array(spikes_IN_OFF[n]), bins=np.arange(0., tsim+binsize, binsize))
                PST_IN_OFF[n,:] += h * (1000. / binsize)

            for n in np.arange(0,len(RCs_ON)):
                h, e = np.histogram(np.array(spikes_RC_ON[n]), bins=np.arange(0., tsim+binsize, binsize))
                PST_RC_ON[n,:] += h * (1000. / binsize)
            for n in np.arange(0,len(RCs_OFF)):
                h, e = np.histogram(np.array(spikes_RC_OFF[n]), bins=np.arange(0., tsim+binsize, binsize))
                PST_RC_OFF[n,:] += h * (1000. / binsize)

            for n in np.arange(0,len(PYs_h_ON)):
                h, e = np.histogram(np.array(spikes_CXPY_h_ON[n]), bins=np.arange(0., tsim+binsize, binsize))
                PST_CXPY_h_ON[n,:] += h * (1000. / binsize)
            for n in np.arange(0,len(PYs_h_OFF)):
                h, e = np.histogram(np.array(spikes_CXPY_h_OFF[n]), bins=np.arange(0., tsim+binsize, binsize))
                PST_CXPY_h_OFF[n,:] += h * (1000. / binsize)

            for n in np.arange(0,len(PYs_v_ON)):
                h, e = np.histogram(np.array(spikes_CXPY_v_ON[n]), bins=np.arange(0., tsim+binsize, binsize))
                PST_CXPY_v_ON[n,:] += h * (1000. / binsize)
            for n in np.arange(0,len(PYs_v_OFF)):
                h, e = np.histogram(np.array(spikes_CXPY_v_OFF[n]), bins=np.arange(0., tsim+binsize, binsize))
                PST_CXPY_v_OFF[n,:] += h * (1000. / binsize)

#            # Save membrane potentials
#            if stim == stim_to_plot:

#                times = [INs_ON[cell_number_IN].tvec,INs_ON[cell_number_IN].tvec,RCs_ON[cell_number].tvec,
#                PYs_h_ON[cell_number].tvec,PYs_v_ON[cell_number].tvec]
#                potentials = [INs_ON[cell_number_IN].vmem[0, :],INs_ON[cell_number_IN].vmem[syn_comps_dist[3], :],
#                RCs_ON[cell_number].vmem[0, :],
#                PYs_h_ON[cell_number].vmem[0, :],PYs_v_ON[cell_number].vmem[0, :]]
#                spikes_arriving = [spikes_ON[cell_number],spikes_ON[cell_number],spikes_ON[cell_number],[0.0],[0.0]]
#                labels = ["IN-ON","IN-ON-distal dendrite[3]","RC-ON","PY-horizontal-ON","PY-vertical-ON"]

#                sim.saveMemPotential(str(stim)+"ON",times,potentials,spikes_arriving,labels)

#                times = [INs_OFF[cell_number_IN].tvec,INs_OFF[cell_number_IN].tvec,RCs_OFF[cell_number].tvec,
#                PYs_h_OFF[cell_number].tvec,PYs_v_OFF[cell_number].tvec]
#                potentials = [INs_OFF[cell_number_IN].vmem[0, :],INs_OFF[cell_number_IN].vmem[syn_comps_dist[3], :],
#                RCs_OFF[cell_number].vmem[0, :],
#                PYs_h_OFF[cell_number].vmem[0, :],PYs_v_OFF[cell_number].vmem[0, :]]
#                spikes_arriving = [spikes_OFF[cell_number],spikes_OFF[cell_number],spikes_OFF[cell_number],[0.0],[0.0]]
#                labels = ["IN-OFF","IN-OFF-distal dendrite[3]","RC-OFF","PY-horizontal-OFF","PY-vertical-OFF"]

#                sim.saveMemPotential(str(stim)+"OFF",times,potentials,spikes_arriving,labels)

            # Release memory
            sim.deleteAll(NEURON_cells_to_sim)
            del NEURON_cells_to_sim

        # Save PSTH matrix (for each stimuli)
        sim.savePST(stim,"IN-ON",PST_IN_ON,type+str(stimulus_type[comb]))
        sim.savePST(stim,"IN-OFF",PST_IN_OFF,type+str(stimulus_type[comb]))
        sim.savePST(stim,"RC-ON",PST_RC_ON,type+str(stimulus_type[comb]))
        sim.savePST(stim,"RC-OFF",PST_RC_OFF,type+str(stimulus_type[comb]))
        sim.savePST(stim,"PY_h-ON",PST_CXPY_h_ON,type+str(stimulus_type[comb]))
        sim.savePST(stim,"PY_h-OFF",PST_CXPY_h_OFF,type+str(stimulus_type[comb]))
        sim.savePST(stim,"PY_v-ON",PST_CXPY_v_ON,type+str(stimulus_type[comb]))
        sim.savePST(stim,"PY_v-OFF",PST_CXPY_v_OFF,type+str(stimulus_type[comb]))

        comb+=1

    return 1


################
##### Main #####
################

def main():

    # Start timer
    if rank==0:
        start_c = time.time()

    # Divide data into chunks
    if rank == 0:
        chunks = [[] for _ in range(size)]
        chunks_exp = [[] for _ in range(size)]
        cc = 0
        for pp1 in p1:
            for pp2 in p2:
                for i, chunk in enumerate(stimulus):
                    chunks[(i+cc*len(stimulus)) % size].append(chunk)
                    chunks_exp[(i+cc*len(stimulus)) % size].append(cc)
                cc+=1

    else:
        chunks = None
        chunks_exp = None

    # Scatter data
    stim = []
    stim = comm.scatter(chunks,root=0)
    stim_exp = []
    stim_exp = comm.scatter(chunks_exp,root=0)
    value_to_return = worker(stim,stim_exp)
    # Gather data (to avoid MPI timed-out errors)
#    results = comm.gather(value_to_return, root=0)

    # End of simulation
    if rank == 0:
        end_c = time.time()
        print("time elapsed (h): %s" % str((end_c - start_c)/3600.0))

if __name__ == '__main__':
    main()

