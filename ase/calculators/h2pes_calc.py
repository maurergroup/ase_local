"""This module defines a derived Aims calculator 
combined with a model potential.
R.J. Maurer
"""

import os
import numpy as np
from ase.units import Hartree, s
from ase.io.aims import write_aims, read_aims
from ase.data import atomic_numbers
import h2pes
import h2agtensor
from ase.calculators.calculator import FileIOCalculator,Calculator

ps = s*1.E-12

class h2pes_calc(FileIOCalculator,Calculator):
    implemented_properties = ['energy', 'forces', 'friction']

    def __init__(self, h2indices=[0,1], restart=None, ignore_bad_restart_file=False,
                 label=os.curdir, atoms=None,
                 **kwargs):

        FileIOCalculator.__init__(self, restart, ignore_bad_restart_file, 
                label, atoms, **kwargs)

        h2pes.pes_init()
        self.h2indices = h2indices
        implemented_properties = ['energy', 'forces', 'friction']
        self.implemented_properties = implemented_properties

        submatrix = np.zeros(len(h2indices)*3, dtype=np.int)
        for i in range(len(h2indices)*3):
            submatrix[i] = h2indices[i/3]*3+i%3
        self.submatrix = submatrix

    def get_potential_energy(self,atoms):
        q = atoms.positions[self.h2indices].flatten()
        #str = ' '
        #for i in q:
            #str+= '{0:16.12f} '.format(i)
        #print 'q ',str
        e = h2pes.pot0(2,q)
        #print 'e ', e
        return e

    def get_forces(self,atoms):
        q = np.array(atoms.positions[self.h2indices].flatten(),dtype=np.float32)
        #str = ' '
        #for i in q:
            #str+= '{0:16.12f} '.format(i)
        #print 'q ',str
        f = np.array(h2pes.dpeshon(2,q)).reshape([2,3])
        #str = ' '
        #for i in f.flatten():
            #str+= '{0:16.12f} '.format(i)
        #print 'f ',str
        forces = np.zeros([len(atoms),3])
        forces[self.h2indices[0],:] = f[0,:]
        forces[self.h2indices[1],:] = f[1,:]
        return -forces
        
    def get_friction_tensor(self,atoms):
        q = np.array(atoms.positions[self.h2indices].flatten(),dtype=np.float32)
        tens = h2agtensor.tensor(q)/ps
        #enfores positive diagonal elements, interpolation error might leave some elements negative
        # E, V = np.linalg.eigh(tens)
        # E_old = E.copy()
        # neg_eig = False
        # for i in range(6):
            # if E[i] <0.0:
                # neg_eig = True
                # E[i] = 0.0
        # if neg_eig:
            # print 'E ', E_old
        # tens = np.dot(V, np.dot(np.diag(E), V.transpose()))
        tensor = np.zeros([3*len(atoms), 3*len(atoms)],dtype=np.float32) 
        for i in range(6):
            for j in range(6):
                tensor[self.submatrix[i],self.submatrix[j]] = tens[i,j]
        return tensor
