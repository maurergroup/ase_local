"""Langevin dynamics class."""


import sys
import numpy as np
from numpy.random import standard_normal
from ase.md.md import MolecularDynamics
from ase.parallel import world
from numpy.linalg import eigh
from copy import copy
from ase.units import kB

class MDEF(MolecularDynamics):
    """Langevin (constant N, V, T) molecular dynamics.

    Usage: MDEF(atoms, dt, temperature, friction)

    atoms
        The list of atoms.
        
    dt
        The time step.

    temperature
        The desired temperature, in energy units.

    friction
        A friction coefficient, typically 1e-4 to 1e-2.

    fixcm
        If True, the position and momentum of the center of mass is
        kept unperturbed.  Default: True.

    The temperature and friction are normally scalars, but in principle one
    quantity per atom could be specified by giving an array.

    This dynamics accesses the atoms using Cartesian coordinates.
    
    ##WE ARE USING THE VANDEN-EIJNDEN INTEGRATOR CPL 429, 310-316 (2006)
    """
    
    _lgv_version = 2  # Helps Asap doing the right thing.  Increment when changing stuff.
    def __init__(self, atoms, timestep, temperature, friction, fixcm=True,
                 trajectory=None, logfile=None, loginterval=1,
                 communicator=world):
        MolecularDynamics.__init__(self, atoms, timestep, trajectory,
                                   logfile, loginterval)
        self.temp = temperature
        
        # self.frict = friction
        self.frict, self.fric_vecs= eigh(friction)
        self.fixcm = fixcm  # will the center of mass be held fixed?
        self.communicator = communicator
        self.updatevars()
        self.random = standard_normal(size=(3*len(atoms)))
        
    def set_temperature(self, temperature):
        self.temp = temperature
        self.updatevars()

    def set_friction(self, friction):
        self.frict, self.fric_vecs= eigh(friction)
        #TODO we might need to check if friction is zero
        # if any([ for f in self.frict])
        # self.frict = friction
        self.updatevars()

    def set_timestep(self, timestep):
        self.dt = timestep
        self.updatevars()

    def updatevars(self):
        dt = self.dt
        self.masses_long = self.masses.repeat(3)
        self.c1 = np.exp(-self.frict*dt/2.)
        self.c2 = np.sqrt((1.-self.c1*self.c1)*self.masses_long*(kB*self.temp))

    def momentum_step(self,p,dt,f,fnew):
        return p+((f+fnew)/2.)*dt

    def position_step(self,r,dt,p,m,f):
        return r+p/m*dt+f/m*(dt*dt/2.)

    def friction_step(self,p,rand):
        return self.c1*p+self.c2*rand

    def step(self, f):
        #Bussi Parinello step with tensor transformation
        dt = self.dt
        f = f
        atoms = self.atoms
        p = self.atoms.get_momenta().flatten()
        r = self.atoms.get_positions().flatten()

        #transform into friction mode space
        p = np.dot(self.fric_vecs.transpose(),p)
        #first friction step
        p = self.friction_step(p,self.random)
        #transform pack into normal space
        p = np.dot(self.fric_vecs,p)
        
        #position update 
        r = self.position_step(r,dt,p,self.masses_long,f.flatten())
        atoms.set_positions(r.reshape([-1,3]))
        fold = f.copy()
        #force update
        f = atoms.get_forces()
        #momentum update
        p = self.momentum_step(p,dt,fold.flatten(),f.flatten())
       
        self.random = standard_normal(size=(3*len(atoms)))
        #transform into friction mode space
        p = np.dot(self.fric_vecs.transpose(),p)
        #second friction step
        p = self.friction_step(p,self.random)
        #transform pack into normal space
        p = np.dot(self.fric_vecs,p)

        atoms.set_momenta(p.reshape([-1,3]))
        return f
