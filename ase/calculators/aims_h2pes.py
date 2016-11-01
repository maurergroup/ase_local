"""This module defines a derived Aims calculator 
combined with a model potential.
R.J. Maurer
"""

import os

import numpy as np

from ase.units import Hartree
from ase.io.aims import write_aims, read_aims
from ase.data import atomic_numbers
import h2pes
from ase.calculators.aims import Aims
from ase.calculators.calculator import FileIOCalculator,Calculator

class Aims_h2pes(Aims,FileIOCalculator,Calculator):
    implemented_properties = ['energy', 'forces', 'friction']

    def __init__(self, h2indices=[0,1], restart=None, ignore_bad_restart_file=False,
                 label=os.curdir, atoms=None, cubes=None, radmul=None,
                 tier=None, **kwargs):
        Aims.__init__(self,restart=restart, ignore_bad_restart_file=\
                ignore_bad_restart_file, label=label,atoms=atoms, \
                cubes=cubes, radmul=radmul, tier=tier, **kwargs)

        h2pes.pes_init()
        self.h2indices = h2indices
        implemented_properties = ['energy', 'forces', 'friction']
        self.implemented_properties = implemented_properties

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
        
    # def get_friction_tensor(self,atoms):
        # if ('calculate_friction' not in self.parameters):
                # raise NotImplementedError
    
        # return self.get_property(self, 'friction', atoms)
