from __future__ import print_function, division
# Copyright (C) 2010, Jesper Friis
# (see accompanying license files for details).

# XXX bravais objects need to hold tolerance eps, *or* temember variant
# from the beginning.
#
# Should they hold a 'cycle' argument or other data to reconstruct a particular
# cell?  (E.g. rotation, niggli transform)
#
# Implement total ordering of Bravais classes 1-14

import numpy as np
from numpy import pi, sin, cos, arccos, sqrt, dot
from numpy.linalg import norm

from ase.utils.arraywrapper import arraylike


@arraylike
class Cell:
    """Parallel epipedal unit cell of up to three dimensions.

    This wraps a 3x3 array whose [i, j]-th element is the jth
    Cartesian coordinate of the ith unit vector.

    Cells of less than three dimensions are represented by placeholder
    unit vectors that are zero."""

    # This overridable variable tells an Atoms object whether atoms.cell
    # and atoms.get_cell() should be a Cell object or an array.
    ase_objtype = 'cell'  # For JSON'ing
    _atoms_use_cellobj = 1#bool(os.environ.get('ASE_DEBUG_CELLOBJ'))

    def __init__(self, array=None, pbc=None):
        if array is None:
            array = np.zeros((3, 3))

        if pbc is None:
            pbc = np.ones(3, bool)

        # We could have lazy attributes for structure (bcc, fcc, ...)
        # and other things.  However this requires making the cell
        # array readonly, else people will modify it and things will
        # be out of synch.
        assert array.shape == (3, 3)
        assert array.dtype == float
        assert pbc.shape == (3,)
        assert pbc.dtype == bool
        self.array = array
        self.pbc = pbc

    def cellpar(self, radians=False):
        return cell_to_cellpar(self.array, radians)

    def todict(self):
        return dict(array=self.array, pbc=self.pbc)

    @property
    def shape(self):
        return self.array.shape

    @classmethod
    def new(cls, cell, pbc=None):
        cell = np.array(cell, float)

        if cell.shape == (3,):
            cell = np.diag(cell)
        elif cell.shape == (6,):
            cell = cellpar_to_cell(cell)
        elif cell.shape != (3, 3):
            raise ValueError('Cell must be length 3 sequence, length 6 '
                             'sequence or 3x3 matrix!')

        return cls(cell, pbc=pbc)

    @classmethod
    def fromcellpar(cls, cellpar, ab_normal=(0, 0, 1), a_direction=None,
                    pbc=None):
        cell = cellpar_to_cell(cellpar, ab_normal, a_direction)
        return Cell(cell, pbc=pbc)

    def bravais(self, eps=2e-4, _niggli_reduce=False, _warn=True):
        # We want to always reduce (so things are as robust as possible)
        # ...or not.  It is not very reliable somehow.
        from ase.geometry.bravais import get_bravais_lattice
        return get_bravais_lattice(self, eps=eps,
                                   _niggli_reduce=_niggli_reduce)

    def bandpath(self, path, npoints=50, eps=2e-4):
        bravais, _ = self.bravais()
        # XXX We need to make sure that the rotation is correct.
        return bravais.bandpath(path, npoints=npoints)

    def complete(self):
        """Convert missing cell vectors into orthogonal unit vectors."""
        return Cell(complete_cell(self.array), self.pbc.copy())

    def copy(self):
        return Cell(self.array.copy(), self.pbc.copy())

    @property
    def dtype(self):
        return self.array.dtype

    @property
    def size(self):
        return self.array.size

    @property
    def T(self):
        return self.array.T

    @property
    def flat(self):
        return self.array.flat

    @property
    def celldim(self):
        # XXX Would name it ndim, but this clashes with ndarray.ndim
        return self.array.any(1).sum()

    @property
    def orthorhombic(self):
        return is_orthorhombic(self.array)

    def lengths(self):
        return np.array([np.linalg.norm(v) for v in self.array])

    def angles(self):
        return self.cellpar()[3:].copy()

    @property
    def ndim(self):
        return self.array.ndim

    def __array__(self, dtype=float):
        if dtype != float:
            raise ValueError('Cannot convert cell to array of type {}'
                             .format(dtype))
        return self.array

    def __bool__(self):
        return bool(self.array.any())

    def __ne__(self, other):
        return self.array != other

    def __eq__(self, other):
        return self.array == other

    __nonzero__ = __bool__

    @property
    def volume(self):
        # Fail or 0 for <3D cells?
        # Definitely 0 since this is currently a property.
        # I think normally it is more convenient just to get zero
        return np.abs(np.linalg.det(self.array))

    def tolist(self):
        return self.array.tolist()

    def scaled_positions(self, positions):
        return np.linalg.solve(self.complete().array.T, positions.T).T

    def cartesian_positions(self, scaled_positions):
        return np.dot(scaled_positions, self.complete().array)

    def reciprocal(self):
        return np.linalg.pinv(self.array).transpose()

    def __repr__(self):
        if self.orthorhombic:
            numbers = self.lengths().tolist()
        else:
            numbers = self.tolist()

        pbc = self.pbc
        if all(pbc):
            pbc = True
        elif not any(pbc):
            pbc = False
        return 'Cell({}, pbc={})'.format(numbers, pbc)

    def niggli_reduce(self):
        from ase.build.tools import niggli_reduce_cell
        cell, op = niggli_reduce_cell(self.array)
        return Cell(cell, self.pbc), op


def unit_vector(x):
    """Return a unit vector in the same direction as x."""
    y = np.array(x, dtype='float')
    return y / norm(y)


def angle(x, y):
    """Return the angle between vectors a and b in degrees."""
    return arccos(dot(x, y) / (norm(x) * norm(y))) * 180. / pi


def cell_to_cellpar(cell, radians=False):
    """Returns the cell parameters [a, b, c, alpha, beta, gamma].

    Angles are in degrees unless radian=True is used.
    """
    lengths = [np.linalg.norm(v) for v in cell]
    angles = []
    for i in range(3):
        j = i - 1
        k = i - 2
        ll = lengths[j] * lengths[k]
        if ll > 1e-16:
            x = np.dot(cell[j], cell[k]) / ll
            angle = 180.0 / pi * arccos(x)
        else:
            angle = 90.0
        angles.append(angle)
    if radians:
        angles = [angle * pi / 180 for angle in angles]
    return np.array(lengths + angles)


def cellpar_to_cell(cellpar, ab_normal=(0, 0, 1), a_direction=None):
    """Return a 3x3 cell matrix from cellpar=[a,b,c,alpha,beta,gamma].

    Angles must be in degrees.

    The returned cell is orientated such that a and b
    are normal to `ab_normal` and a is parallel to the projection of
    `a_direction` in the a-b plane.

    Default `a_direction` is (1,0,0), unless this is parallel to
    `ab_normal`, in which case default `a_direction` is (0,0,1).

    The returned cell has the vectors va, vb and vc along the rows. The
    cell will be oriented such that va and vb are normal to `ab_normal`
    and va will be along the projection of `a_direction` onto the a-b
    plane.

    Example:

    >>> cell = cellpar_to_cell([1, 2, 4, 10, 20, 30], (0, 1, 1), (1, 2, 3))
    >>> np.round(cell, 3)
    array([[ 0.816, -0.408,  0.408],
           [ 1.992, -0.13 ,  0.13 ],
           [ 3.859, -0.745,  0.745]])

    """
    if a_direction is None:
        if np.linalg.norm(np.cross(ab_normal, (1, 0, 0))) < 1e-5:
            a_direction = (0, 0, 1)
        else:
            a_direction = (1, 0, 0)

    # Define rotated X,Y,Z-system, with Z along ab_normal and X along
    # the projection of a_direction onto the normal plane of Z.
    ad = np.array(a_direction)
    Z = unit_vector(ab_normal)
    X = unit_vector(ad - dot(ad, Z) * Z)
    Y = np.cross(Z, X)

    # Express va, vb and vc in the X,Y,Z-system
    alpha, beta, gamma = 90., 90., 90.
    if isinstance(cellpar, (int, float)):
        a = b = c = cellpar
    elif len(cellpar) == 1:
        a = b = c = cellpar[0]
    elif len(cellpar) == 3:
        a, b, c = cellpar
    else:
        a, b, c, alpha, beta, gamma = cellpar

    # Handle orthorhombic cells separately to avoid rounding errors
    eps = 2 * np.spacing(90.0, dtype=np.float64)  # around 1.4e-14
    # alpha
    if abs(abs(alpha) - 90) < eps:
        cos_alpha = 0.0
    else:
        cos_alpha = cos(alpha * pi / 180.0)
    # beta
    if abs(abs(beta) - 90) < eps:
        cos_beta = 0.0
    else:
        cos_beta = cos(beta * pi / 180.0)
    # gamma
    if abs(gamma - 90) < eps:
        cos_gamma = 0.0
        sin_gamma = 1.0
    elif abs(gamma + 90) < eps:
        cos_gamma = 0.0
        sin_gamma = -1.0
    else:
        cos_gamma = cos(gamma * pi / 180.0)
        sin_gamma = sin(gamma * pi / 180.0)

    # Build the cell vectors
    va = a * np.array([1, 0, 0])
    vb = b * np.array([cos_gamma, sin_gamma, 0])
    cx = cos_beta
    cy = (cos_alpha - cos_beta * cos_gamma) / sin_gamma
    cz = sqrt(1. - cx * cx - cy * cy)
    vc = c * np.array([cx, cy, cz])

    # Convert to the Cartesian x,y,z-system
    abc = np.vstack((va, vb, vc))
    T = np.vstack((X, Y, Z))
    cell = dot(abc, T)

    return cell


def metric_from_cell(cell):
    """Calculates the metric matrix from cell, which is given in the
    Cartesian system."""
    cell = np.asarray(cell, dtype=float)
    return np.dot(cell, cell.T)


def crystal_structure_from_cell(cell, eps=2e-4, niggli_reduce=True):
    """Return the crystal structure as a string calculated from the cell.

    Supply a cell (from atoms.get_cell()) and get a string representing
    the crystal structure returned. Works exactly the opposite
    way as ase.dft.kpoints.get_special_points().

    Parameters:

    cell : numpy.array or list
        An array like atoms.get_cell()

    Returns:

    crystal structure : str
        'cubic', 'fcc', 'bcc', 'tetragonal', 'orthorhombic',
        'hexagonal' or 'monoclinic'
    """
    cellpar = cell_to_cellpar(cell)
    abc = cellpar[:3]
    angles = cellpar[3:] / 180 * pi
    a, b, c = abc
    alpha, beta, gamma = angles

    if abc.ptp() < eps and abs(angles - pi / 2).max() < eps:
        return 'cubic'
    elif abc.ptp() < eps and abs(angles - pi / 3).max() < eps:
        return 'fcc'
    elif abc.ptp() < eps and abs(angles - np.arccos(-1 / 3)).max() < eps:
        return 'bcc'
    elif abs(a - b) < eps and abs(angles - pi / 2).max() < eps:
        return 'tetragonal'
    elif abs(angles - pi / 2).max() < eps:
        return 'orthorhombic'
    elif (abs(a - b) < eps and
          (abs(gamma - pi / 3 * 2) < eps or abs(gamma - pi / 3) < eps) and
          abs(angles[:2] - pi / 2).max() < eps):
        return 'hexagonal'
    elif (abs(angles - pi / 2) > eps).sum() == 1:
        return 'monoclinic'
    elif (abc.ptp() < eps and angles.ptp() < eps and
          np.abs(angles).max() < pi / 2):
        return 'rhombohedral type 1'
    elif (abc.ptp() < eps and angles.ptp() < eps and
          np.abs(angles).max() > pi / 2):
        return 'rhombohedral type 2'
    else:
        if niggli_reduce:
            from ase.build.tools import niggli_reduce_cell
            cell, _ = niggli_reduce_cell(cell)
            return crystal_structure_from_cell(cell, niggli_reduce=False)
        raise ValueError('Cannot find crystal structure')


def complete_cell(cell):
    """Calculate complete cell with missing lattice vectors.

    Returns a new 3x3 ndarray.
    """

    cell = np.array(cell, dtype=float)
    missing = np.nonzero(~cell.any(axis=1))[0]

    if len(missing) == 3:
        cell.flat[::4] = 1.0
    if len(missing) == 2:
        # Must decide two vectors:
        i = 3 - missing.sum()
        assert abs(cell[i, missing]).max() < 1e-16, "Don't do that"
        cell[missing, missing] = 1.0
    elif len(missing) == 1:
        i = missing[0]
        cell[i] = np.cross(cell[i - 2], cell[i - 1])
        cell[i] /= np.linalg.norm(cell[i])

    return cell


def is_orthorhombic(cell):
    """Check that cell only has stuff in the diagonal."""
    return not (np.flatnonzero(cell) % 4).any()


def orthorhombic(cell):
    """Return cell as three box dimensions or raise ValueError."""
    if not is_orthorhombic(cell):
        raise ValueError('Not orthorhombic')
    return cell.diagonal().copy()
