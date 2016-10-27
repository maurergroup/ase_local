"""Class for ASE-calculator interface with AQD software."""
import numpy as np
from ase.calculators.calculator import Calculator
import ase.units as units

from aqd.bases import HierarchicalProductBasis, UniformProductBasis
from aqd.operators import *
from aqd.global_variables import METERS_TO_BOHR, HARTREE_TO_EV

from os.path import join as pathjoin

from time import time

BOHR_TO_AA = 1.E10/METERS_TO_BOHR

class aqd(Calculator):
    """ASE calculator.

    This calculator uses a potential calculated with AQD

    It is initialized with a precalculated netcdf databasefile, 
    and an opb.nc basis set file.

    """

    implemented_properties = ['energy', 'forces']

    def __init__(self,coords_object,
            database='database.nc', basis='opb.nc', DATA_DIR='./', **kwargs):
        Calculator.__init__(self, **kwargs)

        #winak coords object
        self.coords = coords_object

        #Initialize coordinate object 
        
        opb = HierarchicalProductBasis([])
        opb.load(pathjoin(DATA_DIR, basis))
        #wfb = UniformProductBasis([])
        #wfb.load(pathjoin(DATA_DIR, 'wfb.nc'))
        self.V = Interpolator(opb)
        self.V.load(pathjoin(DATA_DIR,database))
        #can be evaluated with V(s)
        # V.grad(s)

    def check_state(self, atoms):
        system_changes = Calculator.check_state(self, atoms)
        return system_changes

    def calculate(self, atoms, properties=['energy'], system_changes=None):

        Calculator.calculate(self, atoms, properties, system_changes)
        if len(system_changes)==0:
            pass
        else:
            #t1 = time()
            positions = atoms.get_positions().flatten()
            cell = atoms.get_cell().flatten()
            #calc internal coords
            c = self.coords
            s = c.getS(np.concatenate([positions,cell])) 
            
            ##calc energy
            self.energy = self.V(s) *  HARTREE_TO_EV

            #t2 = time()
            #print t2-t1
            self.positions = positions.copy().reshape(-1,3)
            self.results['energy'] = self.energy
            #if 'forces' in properties:
            ##calc forces
            #forces_int = -self.V.grad(s)
            ##forces_int = self.num_forces(s)
            ###transform forces to cart.
            #self._forces = c.grad_transform(forces_int).reshape(-1,3)*(HARTREE_TO_EV)#/BOHR_TO_AA)
            #t1 = time()
            self._forces = self.num_forces_cart(atoms)
            #t2 = time()
            #print t2-t1
            self._forces -= self._forces.sum(axis=0)
            self.results['forces'] = self._forces *HARTREE_TO_EV
            #self.results['forces'] = self._forces[:-3]
            #self.results['stress'] = self._forces[-3:]
            

    def calculate2(self, atoms, properties=['energy'], system_changes=None):

        Calculator.calculate(self, atoms, properties, system_changes)
        if len(system_changes)==0:
            pass
        else:
            #t1 = time()
            positions = atoms.get_positions().flatten()
            cell = atoms.get_cell().flatten()

            #calc internal coords
            c = self.coords
            s = c.getS(np.concatenate([positions,cell])) 
            
            ##calc energy
            self.energy = self.V(s) *  HARTREE_TO_EV

            self.positions = positions.copy().reshape(-1,3)
            self.results['energy'] = self.energy
            #if 'forces' in properties:
            ##calc forces
            forces_int = -self.V.grad(s)
            ##forces_int = self.num_forces(s)
            ###transform forces to cart.
            self._forces = c.grad_transform(forces_int).reshape(-1,3)*(HARTREE_TO_EV)#/BOHR_TO_AA)
            #self._forces = self.num_forces_cart(atoms)
            #t2 = time()
            #print t2-t1
            self._forces -= self._forces.sum(axis=0)
            #self.results['forces'] = self._forces *HARTREE_TO_EV
            self.results['forces'] = self._forces[:-3]
            self.results['stress'] = self._forces[-3:]
        self.atoms = atoms.copy()

    def num_forces(self, s):

        delta = 0.0001
        #calc internal coords

        forces_int = np.zeros(len(self.coords))
        for i in range(len(forces_int)):
            forces_int[i] = (self.V(s+delta) - self.V(s-delta))/(2.*delta)

        return forces_int
    
    def num_forces_cart(self, atoms):

        positions = atoms.get_positions().flatten()
        cell = atoms.get_cell().flatten()

        #calc internal coords
        c = self.coords
        delta = 0.0001
        #calc internal coords

        forces = np.zeros([len(atoms),3])
        for i in range(len(atoms)):
            for j in range(3):
                tmp = atoms.positions.copy()
                tmp[i,j] += delta 
                x1 = np.concatenate([tmp.flatten(),cell])
                tmp[i,j] -= 2.*delta 
                x2 = np.concatenate([tmp.flatten(),cell])
                forces[i,j] = (self.V(c.getS(x1)) - self.V(c.getS(x2)))/(2.*delta)
        
        return -forces
          
