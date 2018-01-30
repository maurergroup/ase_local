import numpy as np

from ase.md.md import MolecularDynamics


class Euler(MolecularDynamics):
    def __init__(self, atoms, dt, trajectory=None, logfile=None,
                 loginterval=1):
        MolecularDynamics.__init__(self, atoms, dt, trajectory, logfile,
                                   loginterval)
            
    def step(self, f):
        p = self.atoms.get_momenta()
        r = self.atoms.get_positions()
        masses = self.atoms.get_masses()[:, np.newaxis]
        self.atoms.set_positions(r + self.dt * p / masses)
        p += self.dt * f
        self.atoms.set_momenta(p, apply_constraint=False)
        
        f = self.atoms.get_forces(md=True)
        return f
        
        
class FixBondLengths:
    maxiter = 500
    
    def __init__(self, pairs, tolerance=1e-8):
        self.pairs = np.asarray(pairs)
        self.tolerance = tolerance

        self.number_of_adjust_positions_iterations = None
        self.number_of_adjust_momenta_iterations = None
        
    def adjust_positions(self, atoms, new):
        old = atoms.positions
        masses = atoms.get_masses()
        
        for i in range(self.maxiter):
            converged = True
            for a, b in self.pairs:
                d0 = old[a] - old[b]
                d1 = new[a] - new[b]
                m = 1 / (1 / masses[a] + 1 / masses[b])
                x = 0.5 * (np.dot(d0, d0) - np.dot(d1, d1)) / np.dot(d0, d1)
                if abs(x) > self.tolerance:
                    new[a] += x * m / masses[a] * d0
                    new[b] -= x * m / masses[b] * d0
                    converged = False
            if converged:
                break
        else:
            raise RuntimeError('Did not converge')
            
        self.number_of_adjust_positions_iterations = i
                
    def adjust_momenta(self, atoms, p):
        old = atoms.positions
        masses = atoms.get_masses()
        for i in range(self.maxiter):
            converged = True
            for a, b in self.pairs:
                d = old[a] - old[b]
                dv = p[a] / masses[a] - p[b] / masses[b]
                m = 1 / (1 / masses[a] + 1 / masses[b])
                x = -np.dot(dv, d) / np.dot(d, d)
                if abs(x) > self.tolerance:
                    p[a] += x * m * d
                    p[b] -= x * m * d
                    converged = False
            if converged:
                break
        else:
            raise RuntimeError('Did not converge')
            
        self.number_of_adjust_momenta_iterations = i
