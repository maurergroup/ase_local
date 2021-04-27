import numpy as np

from ase.calculators.calculator import Calculator
from ase.neighborlist import neighbor_list


def fcut(r, r0, r1):
    """
    Piecewise quintic C^{2,1} regular polynomial for use as a smooth cutoff.
    Ported from JuLIP.jl, https://github.com/JuliaMolSim/JuLIP.jl
    
    Parameters
    ----------
    r0 - inner cutoff radius
    r1 - outder cutoff radius
    """""
    s = 1.0 - (r - r0) / (r1 - r0)
    return (s >= 1.0) + (((0.0 < s) & (s < 1.0)) *
                         (6.0 * s**5 - 15.0 * s**4 + 10.0 * s**3))


def fcut_d(r, r0, r1):
    """
    Derivative of fcut() function defined above
    """
    s = 1 - (r - r0) / (r1 - r0)
    return -(((0.0 < s) & (s < 1.0)) *
             ((30 * s**4 - 60 * s**3 + 30 * s**2) / (r1 - r0)))


class MorsePotential(Calculator):
    """Morse potential.

    Default values chosen to be similar as Lennard-Jones.
    """

    implemented_properties = ['energy', 'forces']
    default_parameters = {'epsilon': 1.0,
                          'rho0': 6.0,
                          'r0': 1.0,
                          'rcut1': 1.9,
                          'rcut2': 2.7}
    nolabel = True

    def __init__(self, **kwargs):
        """
        Parameters
        ----------
        epsilon: float
          Absolute minimum depth, default 1.0
        r0: float
          Minimum distance, default 1.0
        rho0: float
          Exponential prefactor. The force constant in the potential minimum
          is k = 2 * epsilon * (rho0 / r0)**2, default 6.0
        """
        Calculator.__init__(self, **kwargs)

    def calculate(self, atoms=None, properties=['energy'],
                  system_changes=['positions', 'numbers', 'cell',
                                  'pbc', 'charges', 'magmoms']):
        Calculator.calculate(self, atoms, properties, system_changes)
        epsilon = self.parameters.epsilon
        rho0 = self.parameters.rho0
        r0 = self.parameters.r0
        rcut1 = self.parameters.rcut1 * r0
        rcut2 = self.parameters.rcut2 * r0

        forces = np.zeros((len(self.atoms), 3))
        preF = - 2 * epsilon * rho0 / r0

        i, j, d, D = neighbor_list('ijdD', atoms, rcut2)
        dhat = (D / d[:, None]).T

        expf = np.exp(rho0 * (1.0 - d / r0))
        fc = fcut(d, rcut1, rcut2)

        E = epsilon * expf * (expf - 2)
        dE = preF * expf * (expf - 1) * dhat
        energy = 0.5 * (E * fc).sum()

        F = (dE * fc + E * fcut_d(d, rcut1, rcut2) * dhat).T
        for dim in range(3):
            forces[:, dim] = np.bincount(i, weights=F[:, dim],
                                         minlength=len(atoms))

        self.results['energy'] = energy
        self.results['forces'] = forces


class MorsePotential2:
    """Morse potential2.

    Default values chosen to be similar as Lennard-Jones.
    """
    def __init__(self, a=6.0, D=1.0, r0=1.0):
        self.D = D
        self.a = a
        self.r0 = r0
        self.positions = None

    def update(self, atoms):
        assert not atoms.get_pbc().any()
        if (self.positions is None or
            (self.positions != atoms.get_positions()).any()):
            self.calculate(atoms)

    def get_potential_energy(self, atoms):
        self.update(atoms)
        return self.energy

    def get_forces(self, atoms):
        self.update(atoms)
        return self._forces

    def get_stress(self, atoms):
        return np.zeros((3, 3))

    def calculate(self, atoms):
        positions = atoms.get_positions()
        self.energy = 0.0
        F = 0.0
        self._forces = np.zeros((len(atoms), 3))
        if self.a > 0:
            r = positions[1][0]
        else:
            r = -positions[1][0]
        expf = exp(- self.a * (r - self.r0))
        self.energy += self.D * (1 - expf)**2
        F =  -2 * self.a * self.energy * expf * np.sign(r)
        self._forces[1][0] = F
        self.positions = positions.copy()

        #positions = atoms.get_positions()
        #self.energy = 0.0
        #F = 0.0
        #self._forces = np.zeros((len(atoms), 3))
        #for i1, p1 in enumerate(positions):
            #for i2, p2 in enumerate(positions[:i1]):
                #diff = p2 - p1
                #r = sqrt(np.dot(diff, diff))
                #expf = exp(- self.a * (r - self.r0))
                #self.energy += self.D * (1 - expf)**2
                #F =  -2 * self.a * self.energy * expf * (diff / r)
                #self._forces[i1] -= F
                #self._forces[i2] += F
        #self.positions = positions.copy()

