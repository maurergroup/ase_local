import os
import subprocess
import numpy as np
from ase.calculators.calculator import FileIOCalculator, Calculator


class md_tian(FileIOCalculator):

    """ md_tian IO-calculator class.

    This is a simple I/O ASE interface to the md_tian EMT code
    
    """

    implemented_properties = ['energy', 'forces']
    changes = ['positions', 'numbers', 'cell', 'pbc']

    default_parameters = dict(INPUT_FILE='md_tian.inp',
                            projectile='7 emt515_H.nml ver -1',
                            lattice='7 emt515_Au.nml ver 0',
                              ) #default otherwise rsSCS

    valid_args = ('INPUT_FILE','projectile', 'lattice')

    def __init__(self, restart=None, ignore_bad_restart_file=False,
                 label=None, atoms=None, command=None, **kwargs):
        self.command = command
        Calculator.__init__(self, restart, ignore_bad_restart_file,
                            label, atoms, **kwargs)
        self.calculated = False
        if 'projectile' in kwargs:
            self.set_projectile = True
        else:
            self.set_projectile = False

    def calculate(self, atoms=None, properties=['energy'],
                  system_changes=changes):
        Calculator.calculate(self, atoms, properties, system_changes)
        self.write_input(self.atoms, properties, system_changes)
        if self.command is None:
            raise RuntimeError('Please set $%s environment variable ' %
                               ('ASE_' + self.name.upper() + '_COMMAND') +
                               'or supply the command keyword')
        olddir = os.getcwd()
        try:
            syscall = self.command + ' ' + \
                    self.parameters['INPUT_FILE'] + '> OUT '
            errorcode = subprocess.call(syscall, shell=True)
        finally:
            os.chdir(olddir)

        if errorcode:
            raise RuntimeError('%s returned an error: %d' %
                               (self.name, errorcode))
        self.read_results()

    def write_input(self, atoms, properties=None, system_changes=None):
        """Write input file(s).

        Call this method first in subclasses so that directories are
        created automatically."""

        # Write geometry-file
        atoms.write('md_tian_geometry.in')
        # Write input file
        a = open(self.parameters['INPUT_FILE'], 'w+')
        a.write('start 0\n')
        a.write('ntrajs 1\n')
        a.write('Tsurf 0\n')
        a.write('step 0.0001\n')
        a.write('nsteps 1\n')
        a.write('wstep -2 1\n')
        a.write('Einc 0.0 \n')
        a.write('inclination 0\n')
        a.write('azimuth 0\n')
        # write projectile and lattice, know how many atoms of what
        a.write('lattice    {0} {1} '.format(atoms.get_chemical_symbols()[0], \
                atoms.get_masses()[0]) + self.parameters['lattice']+'\n')
        if self.set_projectile:
            a.write('projectile {0} {1} '.format(atoms.get_chemical_symbols()[-1], \
                    atoms.get_masses()[-1]) + self.parameters['projectile']+'\n')
        a.write('pes emt\n')
        a.write('rep 0 0\n')
        a.write('conf fhiaims md_tian_geometry.in\n')
        a.write('pip -1 bri 6.0\n')
        a.close()
        self.calculated = False

    def read_results(self):
        """Read energy from output file."""

        a = open('traj/mxt_fin00000000.dat', 'r')
        tmp = a.readlines()
        a.close()
        for line in tmp:
            if 'E_pot' in line:
                self.results['energy'] = float(line.split()[-1])
                break
        a = open('conf/mxt_conf00000000_force.dat','r')
        tmp = a.readlines()
        forces = np.zeros([len(tmp),3])
        for i,line in enumerate(tmp):
            forces[i,:] = np.array([float(line.split()[j+1]) for j in range(3)])
        self.results['forces'] = forces 
        self.calculated = True
        os.system('rm -r conf; rm -r traj')

    def get_forces(self, atoms=None):
        self.calculate(atoms)
        return self.results['forces']

    def get_potential_energy(self, atoms=None):
        self.calculate(atoms)
        energy = self.get_property('energy', atoms)
        return energy

