from tight_binding_bismuth_and_antimony_lib import functions as tb
from matplotlib import pyplot as plt
import numpy as np
import random
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# THIS SCRIPT CALCULATES THE BAND STRUCTURE OF BULK Bi-Sb ALLOY (16 atom unit cell with 13 Bi atoms and 3 Sb atoms)

####### Global variables #######

parent_folder = "/home/jdawg/Documents/physics_projects/bismuth_antimony_project/tight_binding/bismuth/working_text_files/"

# Crystal geometry
n_extend = np.array([2, 2, 2])

# Atom types
prefix = "Bi13Sb3_bulk_2x2x2"
atom_types = "BiSb"
atom_positions_file = "atom_positions_Bi_bulk_2atoms"

# Initialize model
model = tb.ModelBulk(prefix, atom_types, n_extend, parent_folder)

model.define_atom_types()
model.define_hopping_matrix(atom_positions_file)
model.write_xsf(prefix + ".xsf", atom_positions_file)

b1 = model.b1_ext
b2 = model.b2_ext
b3 = model.b3_ext

# Define high symmetry points
Gamma = np.zeros(3, dtype=float)
T = 0.5 * (b1 + b2 + b3)
B = 0.75950*b2 + 0.24050*b3 + 0.5*b1
X = 0.5*b2 + 0.5*b1
F = 0.75950*b2 + 0.37025*b3 + 0.37025*b1
L = 0.5*b2

symmetry_points_list = np.array([Gamma, T, B, X, F, Gamma, L])
resolution = 100

model.calculate_band_structure(symmetry_points_list, resolution)
model.plot_band_structure(symmetry_points_list, labels=["Gamma", "T", "B", "X", "F", "Gamma", "L"], E_min=-1, E_max=1)

#width_Å = 0.3
#resolution = 40
#Eb = np.array([[0.0, 0.1], [0.2, 0.3]])

#hkl = np.array([1, 1, 1])
#model.calculate_constant_energy_surfaces(hkl, width_Å, resolution)
#model.plot_constant_energy_surfaces(hkl, width_Å, resolution, Eb_list=Eb, sigma=0.05)
