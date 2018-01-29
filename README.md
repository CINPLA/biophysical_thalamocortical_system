# biophysical_thalamocortical_system
This project is the result of a research work and its associated publication is:

Pablo Martínez-Cañada, Milad Hobbi Mobarhan, Geir Halnes, Marianne Fyhn, Christian Morillas, Francisco Pelayo and Gaute T. Einevoll (2018), *Biophysical Network Modeling of the dLGN Circuit: Effects of Cortical Feedback on Spatial Response Properties of Relay Cells, in PLOS Computational Biology.* https://doi.org/10.1371/journal.pcbi.1005930

The software is licensed under [GNU General Public License](https://github.com/CINPLA/biophysical_thalamocortical_system/blob/master/LICENSE). The source code was developed by [P. Martínez-Cañada](https://github.com/pablomc88) (pablomc@ugr.es), first during his research stay at the [Centre for Integrative Neuroplasticity (CINPLA)](http://www.mn.uio.no/ibv/english/research/sections/fyscell/cinpla/), from the University of Oslo (Norway), and later at Centro de Investigación en Tecnologías de la Información y de las Comunicaciones (CITIC) of the University of Granada (Spain). Some pieces of the code were adapted from the original code by [Thomas Heiberg](https://no.linkedin.com/in/thomasheiberg).

## Prerequisites

The source code has been tested with:

- Python 2.7
- NEURON 7.5 (it may also work with previous versions)
- NEST 2.12.0 (it may also work with previous versions)
- LFPy v1.1.1
- mpi4py 1.3.1

After installing these software packages, compile mod files in folders Geir2011, TC_neuron and cortex_neurons using the command nrnivmodl.

## Directory structure

Scripts for simulation are located in folder thalamocortical. Scripts for plotting and analyzing results are in thalamocortical/plots. Simulation results are saved to thalamocortical/results.

Scripts for simulation are listed below.

- *ganglionCell.py*: Simulation of the retinal ganglion cells.
- *network.py*: Simulation of the thalamocortical network.

Descriptions of neuron models are included in the following files.

- *interneuron.py*: Definition of the properties and synapses of a dLGN interneuron.
- *relayCell.py*: Definition of the properties and synapses of a dLGN relay cell.
- *cortex_PY.py*: Definition of the properties and synapses of a dLGN interneuron.

Other files located in folder 'thalamocortical' are:

- *insertChannels.py*: Function used by neuron models to insert the active ion-channels.
- *simulation.py*: This class provides functions to run the simulation in NEURON and to save/load PSTHs and membrane potentials from neurons.
- *topology.py*: Interface for creating connection masks between neuronal layers.

## Configuration of model parameters and simulation

Some general parameters of the model (e.g., number of cells, visual angle) and parameters of the simulation (e.g., simulation time) can be modified at the beginning of the simulation scripts ganglionCell.py and network.py, between lines “Start of parameters” and “End of parameters”. Parameters to customize the input stimuli are found in ganglionCell.py. 

Simulation scripts can be executed in two different ways. First, using directly the Python interpreter (e.g., python ganglionCell.py). A second way is to use mpirun to run a script in parallel (e.g., `mpirun -np 8 python ganglionCell.py`, in which `-np 8` specifies the number of processes to launch). Before running a simulation of network.py, it is necessary to simulate first both ON and OFF responses of ganglion cells. After simulation, execute the different scripts in thalamocortical/plots to show the simulation results (e.g., `python area_response.py`).
