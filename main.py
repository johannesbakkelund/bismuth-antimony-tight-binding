from matplotlib import pyplot as plt
import numpy as np
from functions import *


import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

####### Global variables #######

# Lattice constants
lattice_const_a = 4.3007
lattice_const_c = 11.2221

# Crystal geometry
n_layers = 60
n_extend_x = 1
n_extend_y = 1
n_atoms_removed = 0

# Atom types (Bi, Sb or BiSb)
atom_types = "Sb"
slab_type = "truncated_bulk"

model = tb.ModelSlab112(lattice_const_a, lattice_const_c, n_layers, n_extend_x, n_extend_y, n_atoms_removed, atom_types, slab_type)
#model.define_atom_types()
#model.define_hopping_matrix()

# Define high symmetry points
Gamma = np.zeros(3, dtype=float)
X1 = model.slab_b1
X2 = model.slab_b2
M = X1 + X2
symmetry_points_list = [Gamma, X2, M, Gamma, X1, M]
resolution = 160

#model.calculate_band_structure(symmetry_points_list, resolution, save_eigvecs=True, save_spin_expectation_values=True)
#model.plot_band_structure(symmetry_points_list, save_plot=False, set_Ef=0.0, load_spin_expectation_values=True)

#spin_expectation_values = model.load_spin_expectation_values("band_structure")
#print(spin_expectation_values[0, 10, :])
