from ase.calculators.qmme import qmme
from ase.calculators.aims import Aims
from ase.calculators.mbdio import mbdio
from ase.io import read


molecule = read('azo.xyz')
slab = molecule

aims_bin = 'mpirun -np 4' + ' /data/michelitsch/devAIMS/bin/aims.150528.mpi.x > aims.out'
basis_set = '/data/michelitsch/devAIMS/fhiaims_doc/species_defaults/light/'
if True:
    QM1 = Aims(command=aims_bin,
               xc='PBE',
               relativistic=('atomic_zora scalar 1.0e-09'),
               spin='none',
               species_dir=basis_set,
               tier=1,
               #k_grid=cluster_kgrid,
               smearing=('gaussian', 0.1),
               sc_accuracy_etot=1e-4,
               sc_accuracy_rho=1e-6,
               sc_accuracy_forces=1e-3,
               output=['hirshfeld'],
               )

    calc_vdw = mbdio(INPUT_FILE='azo.mbd',
                     SETTING_FILE='azo_setting.in',
                     OUTPUT_FILE='azo.log',
                     xc='1',
                     mbd_cfdm_dip_cutoff='200.d0',
                     mbd_scs_dip_cutoff='200.0',
                     mbd_supercell_cutoff='15.d0',
                     mbd_scs_vacuum_axis='.true. .true. .false.',
                     command='/data/michelitsch/students/Martin/MBD_standalone/DFT_MBD_AT_rsSCS.x',
                     mode='TS')

    QMMM = qmme(atoms=slab,
                nqm_regions=1,
                nmm_regions=1,
                qm_calculators=[QM1],
                mm_calculators=[calc_vdw],
                qm_atoms=[[(0, 24)]],
                mm_mode='allatoms')

    print ' +--------** STARTING GEOMETRY OPTIMIZATION **--------+\n'
    slab.set_calculator(QMMM)
    print(slab.get_potential_energy())
    print(QMMM.mm_regions[0].get_potential_energy())
    print(QMMM.mm_regions[0].get_forces())
    #dyn = BFGS(slab, trajectory=sys.argv[0][:-3] + 'opt.traj')
    #dyn.run(fmax=0.025)

    print ' +---------** END OF GEOMETRY OPTIMIZATION **---------+\n'

