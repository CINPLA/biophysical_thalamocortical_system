import numpy as np
import matplotlib.pyplot as plt
import os

## Membrane-potential receptive fields of RCs and PYs to bright and dark spots.

# data path
data_path = "/home/pablo/Desktop/Biophysical_thalamocortical_system/thalamocortical/results/"

def loadArray(path,stim):

    arr = []

    lines = [line.rstrip('\n') for line in open(path, "r")]

    for n in np.arange(0,len(lines)):
        h = lines[int(n)].split(',')
        row = []
        for pos in np.arange(0,len(h)-1):
            row.append( float(h[int(pos)]) )
        arr.append(row)

    return arr

# Loading
trials = 100.0
cell = 4 # cell = 2 (RC-ON), cell = 3 (PY-horizontal-ON), cell = 4 (PY-vertical-ON)
RF_1 = "RF_1"
RF_2 = "RF_2"

times = loadArray(data_path+RF_1+"/times"+str(0.0)+"ON",0.0)[cell]
potential_avg_1 = np.array(loadArray(data_path+RF_1+"/potentials"+str(0.0)+"ON",0.0)[cell])
potential_avg_2 = np.array(loadArray(data_path+RF_2+"/potentials"+str(0.0)+"ON",0.0)[cell])

for stim in np.arange(1.0,trials,1.0):
    potential_avg_1 += np.array(loadArray(data_path+RF_1+"/potentials"+str(stim)+"ON",stim)[cell])
    potential_avg_2 += np.array(loadArray(data_path+RF_2+"/potentials"+str(stim)+"ON",stim)[cell])

plt.plot(times,potential_avg_1[0:-1]/trials)
plt.plot(times,potential_avg_2[0:-1]/trials)

# Save data to file
np.savetxt('tmp/'+'membrane_potential_cell'+str(cell)+"_"+RF_1+'.out', potential_avg_1[0:-1]/trials, delimiter=',')
np.savetxt('tmp/'+'membrane_potential_cell'+str(cell)+"_"+RF_2+'.out', potential_avg_2[0:-1]/trials, delimiter=',')
np.savetxt('tmp/'+'times'+str(cell)+'.out', times, delimiter=',')

# push-pull index
interval_baseline = [int(50.0/0.2), int(100.0/0.2)]
interval_response = [int(130.0/0.2), int(150.0/0.2)]

baseline = np.sum(potential_avg_1[interval_baseline[0]:interval_baseline[1]])\
/(trials*len(potential_avg_1[interval_baseline[0]:interval_baseline[1]]))

print("baseline = %s" % baseline)

P = np.sum(potential_avg_1[interval_response[0]:interval_response[1]])\
/(trials*len(potential_avg_1[interval_response[0]:interval_response[1]])) - baseline
N = np.sum(potential_avg_2[interval_response[0]:interval_response[1]])\
/(trials*len(potential_avg_2[interval_response[0]:interval_response[1]])) -baseline

print("P = %s , N = %s " % (P,N))

ind = np.abs(P + N) / np.max([P,N])

print("push-pull index = ",ind)

plt.show()
