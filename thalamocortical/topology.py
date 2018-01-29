#!/usr/bin/env python
# -*- coding: utf-8 -*-

## Interface for creating connection masks between neuronal layers.

from os.path import join
import numpy as np
import pylab as plt
from time import time
import neuron
import LFPy

nrn = neuron.h

# periodic boundary conditions
periodic = True

class Topology(object):

    def __init__(self):
        print("init")

    # Fit a number of cells 'nCell' into a square
    def fitSquare(self,N,nCell,position):
        conns = []
        nCell = float(nCell)
        for x in np.arange(0,np.ceil(nCell/int(np.sqrt(nCell)))):
            for y in np.arange(0,int(np.sqrt(nCell))):
                # continue last row until complete nCell
                if(len(conns) < nCell):
                    # periodic boundary conditions
                    if (np.remainder(position,N) + y) < N and (int(position/N) + x) < N:
                        conns.append(position + x*N + y)
                    elif (np.remainder(position,N) + y) >= N and (int(position/N) + x) < N:
                        conns.append(position + x*N + (N -1 - np.remainder(position,N)))
                    elif (np.remainder(position,N) + y) < N and (int(position/N) + x) >= N:
                        conns.append(np.remainder(position,N) + y + (N-1)*N)
                    else:
                        conns.append(N*N-1)

        return conns

    # Rectangular mask
    def rectMask(self,N,mask,position):
        conns = []
        mask[0] = float(mask[0])
        mask[1] = float(mask[1])

        start1 = 0
        start2 = 0

        # even
        if(np.remainder(mask[0],2)==0):
            start1 = -int(mask[0]/2) + int(position/N) + 1
        # odd
        else:
            start1 = -int(mask[0]/2) + int(position/N)

        # even
        if(np.remainder(mask[1],2)==0):
            start2 = -int(mask[1]/2) + np.remainder(position,N) + 1
        else:
            start2 = -int(mask[1]/2) + np.remainder(position,N)

        # Find connections
        for x in np.arange(start1,int(mask[0]/2)+1 + int(position/N)):
            for y in np.arange(start2,int(mask[1]/2)+1+np.remainder(position,N)):

                if(x>= 0 and y>=0 and x<N and y<N):
                    conns.append(x*N + y)
                else:
                    # periodic boundary conditions
                    if(periodic):
                        conn = 0.0
                        # Upper left
                        if(x<0 and y<0  and x<N and y<N):
                            conns.append((x+N)*N + (y+N))
                        # Top
                        elif(x< 0 and y>=0  and x<N and y<N):
                            conns.append((x+N)*N + y)
                        # Left
                        elif(x>= 0 and y<0  and x<N and y<N):
                            conns.append(x*N + (y+N))
                        # Upper right
                        elif(x< 0 and y>=0 and x<N and y>=N):
                            conns.append((x+N)*N + y-N)
                        # Right
                        elif(x>= 0 and y>=0 and x<N and y>=N):
                            conns.append(x*N + y-N)
                        # Bottom left
                        elif(x>= 0 and y<0 and x>=N and y<N):
                            conns.append((x-N)*N + y+N)
                        # Bottom
                        elif(x>= 0 and y>=0 and x>=N and y<N):
                            conns.append((x-N)*N + y)
                        # Bottom right
                        elif(x>= 0 and y>=0 and x>=N and y>=N):
                            conns.append((x-N)*N + (y-N))

        return conns

    # Display connectivity pattern
    def showConn(self,N,conns):
        matrix = np.zeros((N,N))

        for n in conns:
            matrix[int(n/N),np.remainder(n,N)]=1.0

        plt.matshow(matrix)
        plt.show()


