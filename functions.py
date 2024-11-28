import numpy as np
import pandas as pd
import time
from scipy import linalg
from joblib import Parallel, delayed
import random

from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.interpolate import UnivariateSpline
import matplotlib.colors as mcol


class ModelCrystal:

    def __init__(self, atom_types, prefix, kwargs=None):
        """
        Defines a Bi, Sb or BiSb (surface_normal e.g. 112 or 111)-slab that is produced by repeating the Liu-Allen geometry (lattice_const_a/c) of 4 atoms n_layers times
        in the z-direction and n_extend_x/y in the x/y-directions. n_layers should be an even number because of how get_edge_participation is
        implemented. n_atoms_removed is for removing spcific atoms from the generated crystal to generate alternative surfaces (truncated bulk
        if n_atoms_removed=0, reconstruction I (II) with 1 (2) atoms removed). atom_types is for defining the crystal geomtry (either Bi or Sb
        from Liu-Allen) and for choosing the working folder for storing/reading files. Supported values of slab_type is currently {truncated_bulk,
        truncated_bulk_BiSb, crystal17 (aka lines with stochiometry Bi13Sb3), reconstruction_typeI, reconstruction_typeII}. kwargs is for defining
        the BiSb hopping params from those of Bi and Sb (factor_1, scale_Bi_Sb = kwargs). After initializing the model, one typically runs
        define_atom_types to generate a list with atom numbers and atom types, it is stored in the atom_types folder. Then one can run
        define_hopping_matrix to generate a list containing all atom numbers, all their neighbouring atom numbers, two numbers that descibe 1st, 2nd
        or 3rd neighbour hopping plus the hopping vector (a1,a2 or a3), and the atom types (Bi or Sb). The hopping matrix is stored as a text file
        in the hopping_matrix_folder. Then one can run calculate_bandstructure along a high symmetry path or calculate_constant_energy_surface to
        solve the hopping matrices for certain values of k, see details in the functions. Lastly one can plot the band structure using plot_band_structure
        or plot_constant_energy_surface, choosing parameters for plotting such as spin exp values or edge participation.
        """

        #############################################################################
        ### Define the crystal lattice parameters and the hopping matrix elements ###
        #############################################################################
        
        
        # Class definition parameters
        self.atom_types = atom_types
        self.prefix= prefix

        # Choose working directory and crystal geometry.
        # Either Bi or Sb (Liu-Allen paper), BiSb not implemented (uses Bi as standard)
        match self.atom_types:
            case "Bi":
                self.parent_folder = "/home/jdawg/Documents/physics_projects/bismuth_antimony_project/tight_binding/bismuth/working_text_files/"
                self.lattice_const_a = 4.5332
                self.lattice_const_c = 11.7967
                self.internal_distance = 0.2341
                self.nearest_neighbor_d = 3.0624
                self.next_nearest_neighbor_d = 3.5120
            case "Sb":
                self.parent_folder = "/home/jdawg/Documents/physics_projects/bismuth_antimony_project/tight_binding/antimony/working_text_files/"
                self.lattice_const_a = 4.3007
                self.lattice_const_c = 11.2221
                self.internal_distance = 0.2336
                self.nearest_neighbor_d = 2.902
                self.next_nearest_neighbor_d = 3.343
            case "BiSb":
                self.parent_folder = "/home/jdawg/Documents/physics_projects/bismuth_antimony_project/tight_binding/bismuth_antimony_alloys/working_text_files/"
                self.lattice_const_a = 4.5332
                self.lattice_const_c = 11.7967
                self.internal_distance = 0.2341
                self.nearest_neighbor_d = 3.0624
                self.next_nearest_neighbor_d = 3.5120


        # Lattice vectors
        self.a1 = np.array([-self.lattice_const_a/2, -np.sqrt(3)*self.lattice_const_a/6, self.lattice_const_c/3])
        self.a2 = np.array([self.lattice_const_a/2, -np.sqrt(3)*self.lattice_const_a/6, self.lattice_const_c/3])
        self.a3 = np.array([0, np.sqrt(3)*self.lattice_const_a/3, self.lattice_const_c/3])
        self.bulk_lattice_vectors = np.array([self.a1, self.a2, self.a3])

        # Define remaining crystal geometry params
        self.angle_alpha = np.arccos(np.dot(self.a1, self.a2) / (np.linalg.norm(self.a1) * np.linalg.norm(self.a2)))
        self.d = np.array([0, 0, 2*self.internal_distance]) * self.lattice_const_c
        self.cos_a = (self.a2-self.d)[0] / np.linalg.norm(self.a2-self.d)
        self.cos_b = (self.a2-self.d)[1] / np.linalg.norm(self.a2-self.d)
        self.cos_c = (self.a2-self.d)[2] / np.linalg.norm(self.a2-self.d)
        self.cos_a_prime = (self.a1+self.a3-self.d)[0] / np.linalg.norm(self.a1+self.a3-self.d)
        self.cos_b_prime = (self.a1+self.a3-self.d)[1] / np.linalg.norm(self.a1+self.a3-self.d)
        self.cos_c_prime = (self.a1+self.a3-self.d)[2] / np.linalg.norm(self.a1+self.a3-self.d)

        # Model parameters Bi
        self.Es_Bi = -10.906
        self.Ep_Bi = -0.486

        # Nearest neighbor hopping terms
        self.V1_ss_sigma_Bi = -0.608
        self.V1_sp_sigma_Bi = 1.320
        self.V1_pp_sigma_Bi = 1.854
        self.V1_pp_pi_Bi = -0.600

        # Next nearest neighbor hopping terms
        self.V2_ss_sigma_Bi = -0.384
        self.V2_sp_sigma_Bi = 0.433
        self.V2_pp_sigma_Bi = 1.394
        self.V2_pp_pi_Bi = -0.344

        # Third neighbor hopping terms
        self.V3_ss_sigma_Bi = 0.000
        self.V3_sp_sigma_Bi = 0.000  # Must be fixed to zero due to symmetry, also for BiSb
        self.V3_pp_sigma_Bi = 0.156
        self.V3_pp_pi_Bi = 0.000

        # Spin orbit coupling parameter λ
        self.spin_orbit_lambda_Bi = 1.5

        # Model parameters Sb
        self.Es_Sb = -10.068
        self.Ep_Sb = -0.926

        # Nearest neighbor hopping terms
        self.V1_ss_sigma_Sb = -0.694
        self.V1_sp_sigma_Sb = 1.554
        self.V1_pp_sigma_Sb = 2.342
        self.V1_pp_pi_Sb = -0.582

        # Next nearest neighbor hopping terms
        self.V2_ss_sigma_Sb = -0.366
        self.V2_sp_sigma_Sb = 0.478
        self.V2_pp_sigma_Sb = 1.418
        self.V2_pp_pi_Sb = -0.393

        # Third neighbor hopping terms
        self.V3_ss_sigma_Sb = 0.000
        self.V3_sp_sigma_Sb = 0.000  # Must be fixed to zero due to symmetry, also for BiSb
        self.V3_pp_sigma_Sb = 0.352
        self.V3_pp_pi_Sb = 0.000

        # Spin orbit coupling parameter λ
        self.spin_orbit_lambda_Sb = 0.6

        # Model parameters Bi-Sb
        if kwargs is None:
            self.factor_1 = -1
            self.scale_Bi_Sb = 1
        else:
            self.factor_1, self.scale_Bi_Sb = kwargs

        # Nearest neighbor hopping terms
        self.V1_ss_sigma_Bi_Sb = self.factor_1*(self.V1_ss_sigma_Bi*self.scale_Bi_Sb + self.V1_ss_sigma_Sb) / (self.scale_Bi_Sb+1)
        self.V1_sp_sigma_Bi_Sb = self.factor_1*(self.V1_sp_sigma_Bi*self.scale_Bi_Sb + self.V1_sp_sigma_Sb) / (self.scale_Bi_Sb+1)
        self.V1_pp_sigma_Bi_Sb = self.factor_1*(self.V1_pp_sigma_Bi*self.scale_Bi_Sb + self.V1_pp_sigma_Sb) / (self.scale_Bi_Sb+1)
        self.V1_pp_pi_Bi_Sb = self.factor_1*(self.V1_pp_pi_Bi*self.scale_Bi_Sb + self.V1_pp_pi_Sb) / (self.scale_Bi_Sb+1)

        # Next nearest neighbor hopping terms
        self.V2_ss_sigma_Bi_Sb = self.factor_1*(self.V2_ss_sigma_Bi*self.scale_Bi_Sb + self.V2_ss_sigma_Sb) / (self.scale_Bi_Sb+1)
        self.V2_sp_sigma_Bi_Sb = self.factor_1*(self.V2_sp_sigma_Bi*self.scale_Bi_Sb + self.V2_sp_sigma_Sb) / (self.scale_Bi_Sb+1)
        self.V2_pp_sigma_Bi_Sb = self.factor_1*(self.V2_pp_sigma_Bi*self.scale_Bi_Sb + self.V2_pp_sigma_Sb) / (self.scale_Bi_Sb+1)
        self.V2_pp_pi_Bi_Sb = self.factor_1*(self.V2_pp_pi_Bi*self.scale_Bi_Sb + self.V2_pp_pi_Sb) / (self.scale_Bi_Sb+1)

        # Third neighbor hopping terms
        self.V3_ss_sigma_Bi_Sb = 0.000
        self.V3_sp_sigma_Bi_Sb = 0.000 # Must be fixed to zero due to symmetry, also for BiSb
        self.V3_pp_sigma_Bi_Sb = self.factor_1*(self.V3_pp_sigma_Bi*self.scale_Bi_Sb + self.V3_pp_sigma_Sb) / (self.scale_Bi_Sb+1)
        self.V3_pp_pi_Bi_Sb = 0.000

        # Number of localized atomic orbitals per atom (including spin)
        self.n_orbitals = 8
        self.electrons_per_atom = 5


    #######################################################################
    ### Define g, the e^(k dot a) for all vectors connecting neighbours ###
    #######################################################################


    
    # g0 - g13
    def get_g0_1(self, k):
        return np.exp(1j * np.dot(k, self.a1-self.d))

    def get_g0_2(self, k):
        return np.exp(1j * np.dot(k, self.a2-self.d))

    def get_g0_3(self, k):
        return np.exp(1j * np.dot(k, self.a3-self.d))

    ###

    def get_g1_1(self, k):
        return -np.exp(1j * np.dot(k, self.a1-self.d)) * self.cos_a

    def get_g1_2(self, k):
        return np.exp(1j * np.dot(k, self.a2-self.d)) * self.cos_a

    ###

    def get_g2_1(self, k):
        return np.exp(1j * np.dot(k, self.a1-self.d)) * self.cos_b

    def get_g2_2(self, k):
        return np.exp(1j * np.dot(k, self.a2-self.d)) * self.cos_b

    def get_g2_3(self, k):
        return -2*np.exp(1j * np.dot(k, self.a3-self.d)) * self.cos_b

    ###

    def get_g3_1(self, k):
        return self.get_g0_1(k) * self.cos_c

    def get_g3_2(self, k):
        return self.get_g0_2(k) * self.cos_c

    def get_g3_3(self, k):
        return self.get_g0_3(k) * self.cos_c

    ###

    def get_g4_1(self, k):
        return np.exp(1j * np.dot(k, self.a1-self.d)) * self.cos_a**2

    def get_g4_2(self, k):
        return np.exp(1j * np.dot(k, self.a2-self.d)) * self.cos_a**2

    ###

    def get_g5_1(self, k):
        return self.get_g0_1(k) - self.get_g4_1(k)

    def get_g5_2(self, k):
        return self.get_g0_2(k) - self.get_g4_2(k)

    def get_g5_3(self, k):
        return self.get_g0_3(k)

    ###

    def get_g6_1(self, k):
        return self.get_g1_1(k) * self.cos_c

    def get_g6_2(self, k):
        return self.get_g1_2(k) * self.cos_c

    ###

    def get_g7_1(self, k):
        return np.exp(1j * np.dot(k, self.a1-self.d)) * self.cos_b**2

    def get_g7_2(self, k):
        return np.exp(1j * np.dot(k, self.a2-self.d)) * self.cos_b**2

    def get_g7_3(self, k):
        return 4*np.exp(1j * np.dot(k, self.a3-self.d)) * self.cos_b**2

    ###

    def get_g8_1(self, k):
        return self.get_g0_1(k) - self.get_g7_1(k)

    def get_g8_2(self, k):
        return self.get_g0_2(k) - self.get_g7_2(k)

    def get_g8_3(self, k):
        return self.get_g0_3(k) - self.get_g7_3(k)

    ###

    def get_g9_1(self, k):
        return self.get_g0_1(k) * self.cos_c**2

    def get_g9_2(self, k):
        return self.get_g0_2(k) * self.cos_c**2

    def get_g9_3(self, k):
        return self.get_g0_3(k) * self.cos_c**2

    ###

    def get_g10_1(self, k):
        return self.get_g0_1(k) * (1 - self.cos_c**2)

    def get_g10_2(self, k):
        return self.get_g0_2(k) * (1 - self.cos_c**2)

    def get_g10_3(self, k):
        return self.get_g0_3(k) * (1 - self.cos_c**2)

    ###

    def get_g11_1(self, k):
        return self.get_g2_1(k) * self.cos_c

    def get_g11_2(self, k):
        return self.get_g2_2(k) * self.cos_c

    def get_g11_3(self, k):
        return self.get_g2_3(k) * self.cos_c

    ###

    def get_g12_1(self, k):
        return self.get_g1_1(k) * self.cos_b

    def get_g12_2(self, k):
        return self.get_g1_2(k) * self.cos_b
        
        
    # g13-g25
    def get_g13_1(self, k):
        return np.exp(1j * np.dot(k, (self.a1+self.a3)-self.d))

    def get_g13_2(self, k):
        return np.exp(1j * np.dot(k, (self.a2+self.a3)-self.d))

    def get_g13_3(self, k):
        return np.exp(1j * np.dot(k, (self.a2+self.a1)-self.d))

    ###
        
    def get_g14_1(self, k):
        return np.exp(1j * np.dot(k, (self.a1+self.a3)-self.d)) * self.cos_a_prime

    def get_g14_2(self, k):
        return -np.exp(1j * np.dot(k, (self.a2+self.a3)-self.d)) * self.cos_a_prime

    ###

    def get_g15_1(self, k):
        return np.exp(1j * np.dot(k, (self.a1+self.a3)-self.d)) * self.cos_b_prime

    def get_g15_2(self, k):
        return np.exp(1j * np.dot(k, (self.a2+self.a3)-self.d)) * self.cos_b_prime

    def get_g15_3(self, k):
        return -2*np.exp(1j * np.dot(k, (self.a2+self.a1)-self.d)) * self.cos_b_prime

    ###

    def get_g16_1(self, k):
        return self.get_g13_1(k) * self.cos_c_prime

    def get_g16_2(self, k):
        return self.get_g13_2(k) * self.cos_c_prime

    def get_g16_3(self, k):
        return self.get_g13_3(k) * self.cos_c_prime

    ###

    def get_g17_1(self, k):
        return np.exp(1j * np.dot(k, (self.a1+self.a3)-self.d)) * self.cos_a_prime**2

    def get_g17_2(self, k):
        return np.exp(1j * np.dot(k, (self.a2+self.a3)-self.d)) * self.cos_a_prime**2

    ###

    def get_g18_1(self, k):
        return self.get_g13_1(k) - self.get_g17_1(k)

    def get_g18_2(self, k):
        return self.get_g13_2(k) - self.get_g17_2(k)

    def get_g18_3(self, k):
        return self.get_g13_3(k)

    ###

    def get_g19_1(self, k):
        return self.get_g14_1(k) * self.cos_c_prime

    def get_g19_2(self, k):
        return self.get_g14_2(k) * self.cos_c_prime

    ###

    def get_g20_1(self, k):
        return np.exp(1j * np.dot(k, (self.a1+self.a3)-self.d)) * self.cos_b_prime**2

    def get_g20_2(self, k):
        return np.exp(1j * np.dot(k, (self.a2+self.a3)-self.d)) * self.cos_b_prime**2

    def get_g20_3(self, k):
        return 4*np.exp(1j * np.dot(k, (self.a2+self.a1)-self.d)) * self.cos_b_prime**2

    ###

    def get_g21_1(self, k):
        return self.get_g13_1(k) - self.get_g20_1(k)

    def get_g21_2(self, k):
        return self.get_g13_2(k) - self.get_g20_2(k)

    def get_g21_3(self, k):
        return self.get_g13_3(k) - self.get_g20_3(k)

    ###

    def get_g22_1(self, k):
        return self.get_g13_1(k) * self.cos_c_prime**2

    def get_g22_2(self, k):
        return self.get_g13_2(k) * self.cos_c_prime**2

    def get_g22_3(self, k):
        return self.get_g13_3(k) * self.cos_c_prime**2

    ###

    def get_g23_1(self, k):
        return self.get_g13_1(k) * (1 - self.cos_c_prime**2)

    def get_g23_2(self, k):
        return self.get_g13_2(k) * (1 - self.cos_c_prime**2)

    def get_g23_3(self, k):
        return self.get_g13_3(k) * (1 - self.cos_c_prime**2)

    ###

    def get_g24_1(self, k):
        return self.get_g15_1(k) * self.cos_c_prime

    def get_g24_2(self, k):
        return self.get_g15_2(k) * self.cos_c_prime

    def get_g24_3(self, k):
        return self.get_g15_3(k) * self.cos_c_prime

    ###

    def get_g25_1(self, k):
        return self.get_g14_1(k) * self.cos_b_prime

    def get_g25_2(self, k):
        return self.get_g14_2(k) * self.cos_b_prime
        
        
    # g26 - g31
    def get_g26_1(self, k):
        return np.exp(1j * np.dot(k, (self.a1-self.a3)))

    def get_g26_2(self, k):
        return np.exp(1j * np.dot(k, (self.a2-self.a3)))

    def get_g26_3(self, k):
        return np.exp(1j * np.dot(k, (self.a1-self.a2)))

    ###

    def get_g27_1(self, k):
        return (-1/2)*np.exp(1j * np.dot(k, (self.a1-self.a3)))

    def get_g27_2(self, k):
        return (1/2)*np.exp(1j * np.dot(k, (self.a2-self.a3)))

    def get_g27_3(self, k):
        return -np.exp(1j * np.dot(k, (self.a1-self.a2)))


    ###

    def get_g28_1(self, k):
        return (-np.sqrt(3) / 2)*np.exp(1j * np.dot(k, (self.a1-self.a3)))

    def get_g28_2(self, k):
        return (-np.sqrt(3) / 2)*np.exp(1j * np.dot(k, (self.a2-self.a3)))


    ###

    def get_g29_1(self, k):
        return (1/4)*np.exp(1j * np.dot(k, (self.a1-self.a3)))

    def get_g29_2(self, k):
        return (1/4)*np.exp(1j * np.dot(k, (self.a2-self.a3)))

    def get_g29_3(self, k):
        return np.exp(1j * np.dot(k, (self.a1-self.a2)))


    ###

    def get_g30_1(self, k):
        return (3/4)*np.exp(1j * np.dot(k, (self.a1-self.a3)))

    def get_g30_2(self, k):
        return (3/4)*np.exp(1j * np.dot(k, (self.a2-self.a3)))


    ###

    def get_g31_1(self, k):
        return (np.sqrt(3)/4)*np.exp(1j * np.dot(k, (self.a1-self.a3)))

    def get_g31_2(self, k):
        return (-np.sqrt(3)/4)*np.exp(1j * np.dot(k, (self.a2-self.a3)))



    ##################################################
    ### Define H, the single-site hopping matrices ###
    ##################################################

    
    
    def get_H_zeros(self, k=None):
        # Return zero matrix #
        return np.zeros((self.n_orbitals, self.n_orbitals), dtype=complex)


    def get_H_onsite_Bi(self, k=None):
        # Initiate zero matrix #
        H_onsite = np.zeros((self.n_orbitals, self.n_orbitals), dtype=complex)
        
        ### Fill upper triangular part of matrix with valuself.Es ###
        
        # Row 0
        H_onsite[0, 0] = self.Es_Bi
        
        # Row 1
        H_onsite[1, 1] = self.Es_Bi
        
        # Row 2
        H_onsite[2, 2] = self.Ep_Bi
        H_onsite[2, 3] = -1j * self.spin_orbit_lambda_Bi / 3
        H_onsite[2, 7] = self.spin_orbit_lambda_Bi / 3
        
        # Row 3
        H_onsite[3, 3] = self.Ep_Bi
        H_onsite[3, 7] = -1j * self.spin_orbit_lambda_Bi / 3
        
        # Row 4
        H_onsite[4, 4] = self.Ep_Bi
        H_onsite[4, 5] = - self.spin_orbit_lambda_Bi / 3
        H_onsite[4, 6] = 1j * self.spin_orbit_lambda_Bi / 3
        
        # Row 5
        H_onsite[5, 5] = self.Ep_Bi
        H_onsite[5, 6] = 1j * self.spin_orbit_lambda_Bi / 3
        
        # Row 6
        H_onsite[6, 6] = self.Ep_Bi
        
        # Row 7
        H_onsite[7, 7] = self.Ep_Bi
        
        # Add the hermitian conjugate of the upper triangular to get hermitian matrix #
        H_onsite += np.transpose(np.conjugate(np.triu(H_onsite, k=1)))
        
        return H_onsite


    def get_H_1st_nn_1_Bi(self, k):
        # Initiate zero matrix #
        H_1st_nn_1 = np.zeros((self.n_orbitals, self.n_orbitals), dtype=complex)
        
        ### Fill matrix with valuself.Es ###
        
        # Row 0
        H_1st_nn_1[0, 0] = self.get_g0_1(k) * self.V1_ss_sigma_Bi
        H_1st_nn_1[0, 2] = self.get_g1_1(k) * self.V1_sp_sigma_Bi
        H_1st_nn_1[0, 3] = self.get_g2_1(k) * self.V1_sp_sigma_Bi
        H_1st_nn_1[0, 4] = self.get_g3_1(k) * self.V1_sp_sigma_Bi
        
        # Row 1
        H_1st_nn_1[1, 1] = self.get_g0_1(k) * self.V1_ss_sigma_Bi
        H_1st_nn_1[1, 5] = self.get_g1_1(k) * self.V1_sp_sigma_Bi
        H_1st_nn_1[1, 6] = self.get_g2_1(k) * self.V1_sp_sigma_Bi
        H_1st_nn_1[1, 7] = self.get_g3_1(k) * self.V1_sp_sigma_Bi
        
        # Row 2
        H_1st_nn_1[2, 0] = -H_1st_nn_1[0, 2]
        H_1st_nn_1[2, 2] = self.get_g4_1(k) * self.V1_pp_sigma_Bi + self.get_g5_1(k) * self.V1_pp_pi_Bi
        H_1st_nn_1[2, 3] = self.get_g12_1(k) * (self.V1_pp_sigma_Bi - self.V1_pp_pi_Bi)
        H_1st_nn_1[2, 4] = self.get_g6_1(k) * (self.V1_pp_sigma_Bi - self.V1_pp_pi_Bi)
        
        # Row 3
        H_1st_nn_1[3, 0] = -H_1st_nn_1[0, 3]
        H_1st_nn_1[3, 2] = H_1st_nn_1[2, 3]
        H_1st_nn_1[3, 3] = self.get_g7_1(k) * self.V1_pp_sigma_Bi + self.get_g8_1(k) * self.V1_pp_pi_Bi
        H_1st_nn_1[3, 4] = self.get_g11_1(k) * (self.V1_pp_sigma_Bi - self.V1_pp_pi_Bi)
        
        # Row 4
        H_1st_nn_1[4, 0] = -H_1st_nn_1[0, 4]
        H_1st_nn_1[4, 2] = H_1st_nn_1[2, 4]
        H_1st_nn_1[4, 3] = H_1st_nn_1[3, 4]
        H_1st_nn_1[4, 4] = self.get_g9_1(k) * self.V1_pp_sigma_Bi + self.get_g10_1(k) * self.V1_pp_pi_Bi
        
        # Row 5
        H_1st_nn_1[5, 1] = -H_1st_nn_1[1, 5]
        H_1st_nn_1[5, 5] = H_1st_nn_1[2, 2]
        H_1st_nn_1[5, 6] = H_1st_nn_1[2, 3]
        H_1st_nn_1[5, 7] = H_1st_nn_1[2, 4]
        
        # Row 6
        H_1st_nn_1[6, 1] = -H_1st_nn_1[1, 6]
        H_1st_nn_1[6, 5] = H_1st_nn_1[3, 2]
        H_1st_nn_1[6, 6] = H_1st_nn_1[3, 3]
        H_1st_nn_1[6, 7] = H_1st_nn_1[3, 4]
        
        # Row 7
        H_1st_nn_1[7, 1] = -H_1st_nn_1[1, 7]
        H_1st_nn_1[7, 5] = H_1st_nn_1[4, 2]
        H_1st_nn_1[7, 6] = H_1st_nn_1[4, 3]
        H_1st_nn_1[7, 7] = H_1st_nn_1[4, 4]
        
        return H_1st_nn_1


    def get_H_1st_nn_2_Bi(self, k):
        # Initiate zero matrix #
        H_1st_nn_2 = np.zeros((self.n_orbitals, self.n_orbitals), dtype=complex)
        
        ### Fill matrix with valuself.Es ###
        
        # Row 0
        H_1st_nn_2[0, 0] = self.get_g0_2(k) * self.V1_ss_sigma_Bi
        H_1st_nn_2[0, 2] = self.get_g1_2(k) * self.V1_sp_sigma_Bi
        H_1st_nn_2[0, 3] = self.get_g2_2(k) * self.V1_sp_sigma_Bi
        H_1st_nn_2[0, 4] = self.get_g3_2(k) * self.V1_sp_sigma_Bi
        
        # Row 1
        H_1st_nn_2[1, 1] = self.get_g0_2(k) * self.V1_ss_sigma_Bi
        H_1st_nn_2[1, 5] = self.get_g1_2(k) * self.V1_sp_sigma_Bi
        H_1st_nn_2[1, 6] = self.get_g2_2(k) * self.V1_sp_sigma_Bi
        H_1st_nn_2[1, 7] = self.get_g3_2(k) * self.V1_sp_sigma_Bi
        
        # Row 2
        H_1st_nn_2[2, 0] = -H_1st_nn_2[0, 2]
        H_1st_nn_2[2, 2] = self.get_g4_2(k) * self.V1_pp_sigma_Bi + self.get_g5_2(k) * self.V1_pp_pi_Bi
        H_1st_nn_2[2, 3] = self.get_g12_2(k) * (self.V1_pp_sigma_Bi - self.V1_pp_pi_Bi)
        H_1st_nn_2[2, 4] = self.get_g6_2(k) * (self.V1_pp_sigma_Bi - self.V1_pp_pi_Bi)
        
        # Row 3
        H_1st_nn_2[3, 0] = -H_1st_nn_2[0, 3]
        H_1st_nn_2[3, 2] = H_1st_nn_2[2, 3]
        H_1st_nn_2[3, 3] = self.get_g7_2(k) * self.V1_pp_sigma_Bi + self.get_g8_2(k) * self.V1_pp_pi_Bi
        H_1st_nn_2[3, 4] = self.get_g11_2(k) * (self.V1_pp_sigma_Bi - self.V1_pp_pi_Bi)
        
        # Row 4
        H_1st_nn_2[4, 0] = -H_1st_nn_2[0, 4]
        H_1st_nn_2[4, 2] = H_1st_nn_2[2, 4]
        H_1st_nn_2[4, 3] = H_1st_nn_2[3, 4]
        H_1st_nn_2[4, 4] = self.get_g9_2(k) * self.V1_pp_sigma_Bi + self.get_g10_2(k) * self.V1_pp_pi_Bi
        
        # Row 5
        H_1st_nn_2[5, 1] = -H_1st_nn_2[1, 5]
        H_1st_nn_2[5, 5] = H_1st_nn_2[2, 2]
        H_1st_nn_2[5, 6] = H_1st_nn_2[2, 3]
        H_1st_nn_2[5, 7] = H_1st_nn_2[2, 4]
        
        # Row 6
        H_1st_nn_2[6, 1] = -H_1st_nn_2[1, 6]
        H_1st_nn_2[6, 5] = H_1st_nn_2[3, 2]
        H_1st_nn_2[6, 6] = H_1st_nn_2[3, 3]
        H_1st_nn_2[6, 7] = H_1st_nn_2[3, 4]
        
        # Row 7
        H_1st_nn_2[7, 1] = -H_1st_nn_2[1, 7]
        H_1st_nn_2[7, 5] = H_1st_nn_2[4, 2]
        H_1st_nn_2[7, 6] = H_1st_nn_2[4, 3]
        H_1st_nn_2[7, 7] = H_1st_nn_2[4, 4]
        
        return H_1st_nn_2


    def get_H_1st_nn_3_Bi(self, k):
        # Initiate zero matrix #
        H_1st_nn_3 = np.zeros((self.n_orbitals, self.n_orbitals), dtype=complex)
        
        ### Fill matrix with valuself.Es ###
        
        # Row 0
        H_1st_nn_3[0, 0] = self.get_g0_3(k) * self.V1_ss_sigma_Bi
        H_1st_nn_3[0, 3] = self.get_g2_3(k) * self.V1_sp_sigma_Bi
        H_1st_nn_3[0, 4] = self.get_g3_3(k) * self.V1_sp_sigma_Bi
        
        # Row 1
        H_1st_nn_3[1, 1] = self.get_g0_3(k) * self.V1_ss_sigma_Bi
        H_1st_nn_3[1, 6] = self.get_g2_3(k) * self.V1_sp_sigma_Bi
        H_1st_nn_3[1, 7] = self.get_g3_3(k) * self.V1_sp_sigma_Bi
        
        # Row 2
        H_1st_nn_3[2, 0] = -H_1st_nn_3[0, 2]
        H_1st_nn_3[2, 2] = self.get_g5_3(k) * self.V1_pp_pi_Bi
        
        # Row 3
        H_1st_nn_3[3, 0] = -H_1st_nn_3[0, 3]
        H_1st_nn_3[3, 2] = H_1st_nn_3[2, 3]
        H_1st_nn_3[3, 3] = self.get_g7_3(k) * self.V1_pp_sigma_Bi + self.get_g8_3(k) * self.V1_pp_pi_Bi
        H_1st_nn_3[3, 4] = self.get_g11_3(k) * (self.V1_pp_sigma_Bi - self.V1_pp_pi_Bi)
        
        # Row 4
        H_1st_nn_3[4, 0] = -H_1st_nn_3[0, 4]
        H_1st_nn_3[4, 2] = H_1st_nn_3[2, 4]
        H_1st_nn_3[4, 3] = H_1st_nn_3[3, 4]
        H_1st_nn_3[4, 4] = self.get_g9_3(k) * self.V1_pp_sigma_Bi + self.get_g10_3(k) * self.V1_pp_pi_Bi
        
        # Row 5
        H_1st_nn_3[5, 1] = -H_1st_nn_3[1, 5]
        H_1st_nn_3[5, 5] = H_1st_nn_3[2, 2]
        H_1st_nn_3[5, 6] = H_1st_nn_3[2, 3]
        H_1st_nn_3[5, 7] = H_1st_nn_3[2, 4]
        
        # Row 6
        H_1st_nn_3[6, 1] = -H_1st_nn_3[1, 6]
        H_1st_nn_3[6, 5] = H_1st_nn_3[3, 2]
        H_1st_nn_3[6, 6] = H_1st_nn_3[3, 3]
        H_1st_nn_3[6, 7] = H_1st_nn_3[3, 4]
        
        # Row 7
        H_1st_nn_3[7, 1] = -H_1st_nn_3[1, 7]
        H_1st_nn_3[7, 5] = H_1st_nn_3[4, 2]
        H_1st_nn_3[7, 6] = H_1st_nn_3[4, 3]
        H_1st_nn_3[7, 7] = H_1st_nn_3[4, 4]
        
        return H_1st_nn_3


    def get_H_2nd_nn_1_Bi(self, k):
        # Initiate zero matrix #
        H_2nd_nn_1 = np.zeros((self.n_orbitals, self.n_orbitals), dtype=complex)
        
        ### Fill matrix with valuself.Es ###
        
        # Row 0
        H_2nd_nn_1[0, 0] = self.get_g13_1(k) * self.V2_ss_sigma_Bi
        H_2nd_nn_1[0, 2] = self.get_g14_1(k) * self.V2_sp_sigma_Bi
        H_2nd_nn_1[0, 3] = self.get_g15_1(k) * self.V2_sp_sigma_Bi
        H_2nd_nn_1[0, 4] = self.get_g16_1(k) * self.V2_sp_sigma_Bi
        
        # Row 1
        H_2nd_nn_1[1, 1] = self.get_g13_1(k) * self.V2_ss_sigma_Bi
        H_2nd_nn_1[1, 5] = self.get_g14_1(k) * self.V2_sp_sigma_Bi
        H_2nd_nn_1[1, 6] = self.get_g15_1(k) * self.V2_sp_sigma_Bi
        H_2nd_nn_1[1, 7] = self.get_g16_1(k) * self.V2_sp_sigma_Bi
        
        # Row 2
        H_2nd_nn_1[2, 0] = -H_2nd_nn_1[0, 2]
        H_2nd_nn_1[2, 2] = self.get_g17_1(k) * self.V2_pp_sigma_Bi + self.get_g18_1(k) * self.V2_pp_pi_Bi
        H_2nd_nn_1[2, 3] = self.get_g25_1(k) * (self.V2_pp_sigma_Bi - self.V2_pp_pi_Bi)
        H_2nd_nn_1[2, 4] = self.get_g19_1(k) * (self.V2_pp_sigma_Bi - self.V2_pp_pi_Bi)
        
        # Row 3
        H_2nd_nn_1[3, 0] = -H_2nd_nn_1[0, 3]
        H_2nd_nn_1[3, 2] = H_2nd_nn_1[2, 3]
        H_2nd_nn_1[3, 3] = self.get_g20_1(k) * self.V2_pp_sigma_Bi + self.get_g21_1(k) * self.V2_pp_pi_Bi
        H_2nd_nn_1[3, 4] = self.get_g24_1(k) * (self.V2_pp_sigma_Bi - self.V2_pp_pi_Bi)
        
        # Row 4
        H_2nd_nn_1[4, 0] = -H_2nd_nn_1[0, 4]
        H_2nd_nn_1[4, 2] = H_2nd_nn_1[2, 4]
        H_2nd_nn_1[4, 3] = H_2nd_nn_1[3, 4]
        H_2nd_nn_1[4, 4] = self.get_g22_1(k) * self.V2_pp_sigma_Bi + self.get_g23_1(k) * self.V2_pp_pi_Bi
        
        # Row 5
        H_2nd_nn_1[5, 1] = -H_2nd_nn_1[1, 5]
        H_2nd_nn_1[5, 5] = H_2nd_nn_1[2, 2]
        H_2nd_nn_1[5, 6] = H_2nd_nn_1[2, 3]
        H_2nd_nn_1[5, 7] = H_2nd_nn_1[2, 4]
        
        # Row 6
        H_2nd_nn_1[6, 1] = -H_2nd_nn_1[1, 6]
        H_2nd_nn_1[6, 5] = H_2nd_nn_1[3, 2]
        H_2nd_nn_1[6, 6] = H_2nd_nn_1[3, 3]
        H_2nd_nn_1[6, 7] = H_2nd_nn_1[3, 4]
        
        # Row 7
        H_2nd_nn_1[7, 1] = -H_2nd_nn_1[1, 7]
        H_2nd_nn_1[7, 5] = H_2nd_nn_1[4, 2]
        H_2nd_nn_1[7, 6] = H_2nd_nn_1[4, 3]
        H_2nd_nn_1[7, 7] = H_2nd_nn_1[4, 4]
        
        return H_2nd_nn_1


    def get_H_2nd_nn_2_Bi(self, k):
        # Initiate zero matrix #
        H_2nd_nn_2 = np.zeros((self.n_orbitals, self.n_orbitals), dtype=complex)
        
        ### Fill matrix with valuself.Es ###
        
        # Row 0
        H_2nd_nn_2[0, 0] = self.get_g13_2(k) * self.V2_ss_sigma_Bi
        H_2nd_nn_2[0, 2] = self.get_g14_2(k) * self.V2_sp_sigma_Bi
        H_2nd_nn_2[0, 3] = self.get_g15_2(k) * self.V2_sp_sigma_Bi
        H_2nd_nn_2[0, 4] = self.get_g16_2(k) * self.V2_sp_sigma_Bi
        
        # Row 1
        H_2nd_nn_2[1, 1] = self.get_g13_2(k) * self.V2_ss_sigma_Bi
        H_2nd_nn_2[1, 5] = self.get_g14_2(k) * self.V2_sp_sigma_Bi
        H_2nd_nn_2[1, 6] = self.get_g15_2(k) * self.V2_sp_sigma_Bi
        H_2nd_nn_2[1, 7] = self.get_g16_2(k) * self.V2_sp_sigma_Bi
        
        # Row 2
        H_2nd_nn_2[2, 0] = -H_2nd_nn_2[0, 2]
        H_2nd_nn_2[2, 2] = self.get_g17_2(k) * self.V2_pp_sigma_Bi + self.get_g18_2(k) * self.V2_pp_pi_Bi
        H_2nd_nn_2[2, 3] = self.get_g25_2(k) * (self.V2_pp_sigma_Bi - self.V2_pp_pi_Bi)
        H_2nd_nn_2[2, 4] = self.get_g19_2(k) * (self.V2_pp_sigma_Bi - self.V2_pp_pi_Bi)
        
        # Row 3
        H_2nd_nn_2[3, 0] = -H_2nd_nn_2[0, 3]
        H_2nd_nn_2[3, 2] = H_2nd_nn_2[2, 3]
        H_2nd_nn_2[3, 3] = self.get_g20_2(k) * self.V2_pp_sigma_Bi + self.get_g21_2(k) * self.V2_pp_pi_Bi
        H_2nd_nn_2[3, 4] = self.get_g24_2(k) * (self.V2_pp_sigma_Bi - self.V2_pp_pi_Bi)
        
        # Row 4
        H_2nd_nn_2[4, 0] = -H_2nd_nn_2[0, 4]
        H_2nd_nn_2[4, 2] = H_2nd_nn_2[2, 4]
        H_2nd_nn_2[4, 3] = H_2nd_nn_2[3, 4]
        H_2nd_nn_2[4, 4] = self.get_g22_2(k) * self.V2_pp_sigma_Bi + self.get_g23_2(k) * self.V2_pp_pi_Bi
        
        # Row 5
        H_2nd_nn_2[5, 1] = -H_2nd_nn_2[1, 5]
        H_2nd_nn_2[5, 5] = H_2nd_nn_2[2, 2]
        H_2nd_nn_2[5, 6] = H_2nd_nn_2[2, 3]
        H_2nd_nn_2[5, 7] = H_2nd_nn_2[2, 4]
        
        # Row 6
        H_2nd_nn_2[6, 1] = -H_2nd_nn_2[1, 6]
        H_2nd_nn_2[6, 5] = H_2nd_nn_2[3, 2]
        H_2nd_nn_2[6, 6] = H_2nd_nn_2[3, 3]
        H_2nd_nn_2[6, 7] = H_2nd_nn_2[3, 4]
        
        # Row 7
        H_2nd_nn_2[7, 1] = -H_2nd_nn_2[1, 7]
        H_2nd_nn_2[7, 5] = H_2nd_nn_2[4, 2]
        H_2nd_nn_2[7, 6] = H_2nd_nn_2[4, 3]
        H_2nd_nn_2[7, 7] = H_2nd_nn_2[4, 4]
        
        return H_2nd_nn_2


    def get_H_2nd_nn_3_Bi(self, k):
        # Initiate zero matrix #
        H_2nd_nn_3 = np.zeros((self.n_orbitals, self.n_orbitals), dtype=complex)
        
        ### Fill matrix with valuself.Es ###
        
        # Row 0
        H_2nd_nn_3[0, 0] = self.get_g13_3(k) * self.V2_ss_sigma_Bi
        H_2nd_nn_3[0, 3] = self.get_g15_3(k) * self.V2_sp_sigma_Bi
        H_2nd_nn_3[0, 4] = self.get_g16_3(k) * self.V2_sp_sigma_Bi
        
        # Row 1
        H_2nd_nn_3[1, 1] = self.get_g13_3(k) * self.V2_ss_sigma_Bi
        H_2nd_nn_3[1, 6] = self.get_g15_3(k) * self.V2_sp_sigma_Bi
        H_2nd_nn_3[1, 7] = self.get_g16_3(k) * self.V2_sp_sigma_Bi
        
        # Row 2
        H_2nd_nn_3[2, 0] = -H_2nd_nn_3[0, 2]
        H_2nd_nn_3[2, 2] = self.get_g18_3(k) * self.V2_pp_pi_Bi
        
        # Row 3
        H_2nd_nn_3[3, 0] = -H_2nd_nn_3[0, 3]
        H_2nd_nn_3[3, 2] = H_2nd_nn_3[2, 3]
        H_2nd_nn_3[3, 3] = self.get_g20_3(k) * self.V2_pp_sigma_Bi + self.get_g21_3(k) * self.V2_pp_pi_Bi
        H_2nd_nn_3[3, 4] = self.get_g24_3(k) * (self.V2_pp_sigma_Bi - self.V2_pp_pi_Bi)
        
        # Row 4
        H_2nd_nn_3[4, 0] = -H_2nd_nn_3[0, 4]
        H_2nd_nn_3[4, 2] = H_2nd_nn_3[2, 4]
        H_2nd_nn_3[4, 3] = H_2nd_nn_3[3, 4]
        H_2nd_nn_3[4, 4] = self.get_g22_3(k) * self.V2_pp_sigma_Bi + self.get_g23_3(k) * self.V2_pp_pi_Bi
        
        # Row 5
        H_2nd_nn_3[5, 1] = -H_2nd_nn_3[1, 5]
        H_2nd_nn_3[5, 5] = H_2nd_nn_3[2, 2]
        H_2nd_nn_3[5, 6] = H_2nd_nn_3[2, 3]
        H_2nd_nn_3[5, 7] = H_2nd_nn_3[2, 4]
        
        # Row 6
        H_2nd_nn_3[6, 1] = -H_2nd_nn_3[1, 6]
        H_2nd_nn_3[6, 5] = H_2nd_nn_3[3, 2]
        H_2nd_nn_3[6, 6] = H_2nd_nn_3[3, 3]
        H_2nd_nn_3[6, 7] = H_2nd_nn_3[3, 4]
        
        # Row 7
        H_2nd_nn_3[7, 1] = -H_2nd_nn_3[1, 7]
        H_2nd_nn_3[7, 5] = H_2nd_nn_3[4, 2]
        H_2nd_nn_3[7, 6] = H_2nd_nn_3[4, 3]
        H_2nd_nn_3[7, 7] = H_2nd_nn_3[4, 4]
        
        return H_2nd_nn_3


    def get_H_3rd_nn_1_Bi(self, k):
        # Initiate zero matrix #
        H_3rd_nn_1 = np.zeros((self.n_orbitals, self.n_orbitals), dtype=complex)
        
        ### Fill upper triangular part of matrix with valuself.Es ###
        
        # Row 0
        H_3rd_nn_1[0, 0] = self.get_g26_1(k) * self.V3_ss_sigma_Bi
        H_3rd_nn_1[0, 2] = self.get_g27_1(k) * self.V3_sp_sigma_Bi
        H_3rd_nn_1[0, 3] = self.get_g28_1(k) * self.V3_sp_sigma_Bi
        
        # Row 1
        H_3rd_nn_1[1, 1] = self.get_g26_1(k) * self.V3_ss_sigma_Bi
        H_3rd_nn_1[1, 5] = self.get_g27_1(k) * self.V3_sp_sigma_Bi
        H_3rd_nn_1[1, 6] = self.get_g28_1(k) * self.V3_sp_sigma_Bi
        
        # Row 2
        H_3rd_nn_1[2, 2] = self.get_g29_1(k) * self.V3_pp_sigma_Bi + self.get_g30_1(k) * self.V3_pp_pi_Bi
        H_3rd_nn_1[2, 3] = self.get_g31_1(k) * (self.V3_pp_sigma_Bi - self.V3_pp_pi_Bi)
        
        # Row 3
        H_3rd_nn_1[3, 3] = self.get_g30_1(k) * self.V3_pp_sigma_Bi + self.get_g29_1(k) * self.V3_pp_pi_Bi
        
        # Row 4
        H_3rd_nn_1[4, 4] = self.get_g26_1(k) * self.V3_pp_pi_Bi
        
        # Row 5
        H_3rd_nn_1[5, 5] = self.get_g29_1(k) * self.V3_pp_sigma_Bi + self.get_g30_1(k) * self.V3_pp_pi_Bi
        H_3rd_nn_1[5, 6] = self.get_g31_1(k) * (self.V3_pp_sigma_Bi - self.V3_pp_pi_Bi)
        
        # Row 6
        H_3rd_nn_1[6, 6] = self.get_g30_1(k) * self.V3_pp_sigma_Bi + self.get_g29_1(k) * self.V3_pp_pi_Bi
        
        # Row 7
        H_3rd_nn_1[7, 7] = self.get_g26_1(k) * self.V3_pp_pi_Bi
        
        # Add the hermitian conjugate of the upper triangular to get hermitian matrix #
        H_3rd_nn_1 += np.transpose(np.triu(H_3rd_nn_1, k=1))
        
        return H_3rd_nn_1


    def get_H_3rd_nn_2_Bi(self, k):
        # Initiate zero matrix #
        H_3rd_nn_2 = np.zeros((self.n_orbitals, self.n_orbitals), dtype=complex)
        
        ### Fill upper triangular part of matrix with valuself.Es ###
        
        # Row 0
        H_3rd_nn_2[0, 0] = self.get_g26_2(k) * self.V3_ss_sigma_Bi
        H_3rd_nn_2[0, 2] = self.get_g27_2(k) * self.V3_sp_sigma_Bi
        H_3rd_nn_2[0, 3] = self.get_g28_2(k) * self.V3_sp_sigma_Bi
        
        # Row 1
        H_3rd_nn_2[1, 1] = self.get_g26_2(k) * self.V3_ss_sigma_Bi
        H_3rd_nn_2[1, 5] = self.get_g27_2(k) * self.V3_sp_sigma_Bi
        H_3rd_nn_2[1, 6] = self.get_g28_2(k) * self.V3_sp_sigma_Bi
        
        # Row 2
        H_3rd_nn_2[2, 2] = self.get_g29_2(k) * self.V3_pp_sigma_Bi + self.get_g30_2(k) * self.V3_pp_pi_Bi
        H_3rd_nn_2[2, 3] = self.get_g31_2(k) * (self.V3_pp_sigma_Bi - self.V3_pp_pi_Bi)
        
        # Row 3
        H_3rd_nn_2[3, 3] = self.get_g30_2(k) * self.V3_pp_sigma_Bi + self.get_g29_2(k) * self.V3_pp_pi_Bi
        
        # Row 4
        H_3rd_nn_2[4, 4] = self.get_g26_2(k) * self.V3_pp_pi_Bi
        
        # Row 5
        H_3rd_nn_2[5, 5] = self.get_g29_2(k) * self.V3_pp_sigma_Bi + self.get_g30_2(k) * self.V3_pp_pi_Bi
        H_3rd_nn_2[5, 6] = self.get_g31_2(k) * (self.V3_pp_sigma_Bi - self.V3_pp_pi_Bi)
        
        # Row 6
        H_3rd_nn_2[6, 6] = self.get_g30_2(k) * self.V3_pp_sigma_Bi + self.get_g29_2(k) * self.V3_pp_pi_Bi
        
        # Row 7
        H_3rd_nn_2[7, 7] = self.get_g26_2(k) * self.V3_pp_pi_Bi
        
        # Add the hermitian conjugate of the upper triangular to get hermition matrix #
        H_3rd_nn_2 += np.transpose(np.triu(H_3rd_nn_2, k=1))
        
        return H_3rd_nn_2


    def get_H_3rd_nn_3_Bi(self, k):
        # Initiate zero matrix #
        H_3rd_nn_3 = np.zeros((self.n_orbitals, self.n_orbitals), dtype=complex)
        
        ### Fill upper triangular part of matrix with valuself.Es ###
        
        # Row 0
        H_3rd_nn_3[0, 0] = self.get_g26_3(k) * self.V3_ss_sigma_Bi
        H_3rd_nn_3[0, 2] = self.get_g27_3(k) * self.V3_sp_sigma_Bi
        
        # Row 1
        H_3rd_nn_3[1, 1] = self.get_g26_3(k) * self.V3_ss_sigma_Bi
        H_3rd_nn_3[1, 5] = self.get_g27_3(k) * self.V3_sp_sigma_Bi
        
        # Row 2
        H_3rd_nn_3[2, 2] = self.get_g29_3(k) * self.V3_pp_sigma_Bi
        
        # Row 3
        H_3rd_nn_3[3, 3] = self.get_g29_3(k) * self.V3_pp_pi_Bi
        
        # Row 4
        H_3rd_nn_3[4, 4] = self.get_g26_3(k) * self.V3_pp_pi_Bi
        
        # Row 5
        H_3rd_nn_3[5, 5] = self.get_g29_3(k) * self.V3_pp_sigma_Bi
        
        # Row 6
        H_3rd_nn_3[6, 6] = self.get_g29_3(k) * self.V3_pp_pi_Bi
        
        # Row 7
        H_3rd_nn_3[7, 7] = self.get_g26_3(k) * self.V3_pp_pi_Bi
        
        # Add the hermitian conjugate of the upper triangular to get hermition matrix #
        H_3rd_nn_3 += np.transpose(np.triu(H_3rd_nn_3, k=1))
        
        return H_3rd_nn_3


    def get_H_all_nn_Bi(self):
        return [
            [self.get_H_1st_nn_1_Bi, self.get_H_1st_nn_2_Bi, self.get_H_1st_nn_3_Bi],
            [self.get_H_2nd_nn_1_Bi, self.get_H_2nd_nn_2_Bi, self.get_H_2nd_nn_3_Bi],
            [self.get_H_3rd_nn_1_Bi, self.get_H_3rd_nn_2_Bi, self.get_H_3rd_nn_3_Bi]
            ]

            
    def get_H_onsite_Sb(self, k=None):
        # Initiate zero matrix #
        H_onsite = np.zeros((self.n_orbitals, self.n_orbitals), dtype=complex)
        
        ### Fill upper triangular part of matrix with valuself.Es ###
        
        # Row 0
        H_onsite[0, 0] = self.Es_Sb
        
        # Row 1
        H_onsite[1, 1] = self.Es_Sb
        
        # Row 2
        H_onsite[2, 2] = self.Ep_Sb
        H_onsite[2, 3] = -1j * self.spin_orbit_lambda_Sb / 3
        H_onsite[2, 7] = self.spin_orbit_lambda_Sb / 3
        
        # Row 3
        H_onsite[3, 3] = self.Ep_Sb
        H_onsite[3, 7] = -1j * self.spin_orbit_lambda_Sb / 3
        
        # Row 4
        H_onsite[4, 4] = self.Ep_Sb
        H_onsite[4, 5] = - self.spin_orbit_lambda_Sb / 3
        H_onsite[4, 6] = 1j * self.spin_orbit_lambda_Sb / 3
        
        # Row 5
        H_onsite[5, 5] = self.Ep_Sb
        H_onsite[5, 6] = 1j * self.spin_orbit_lambda_Sb / 3
        
        # Row 6
        H_onsite[6, 6] = self.Ep_Sb
        
        # Row 7
        H_onsite[7, 7] = self.Ep_Sb
        
        # Add the hermitian conjugate of the upper triangular to get hermitian matrix #
        H_onsite += np.transpose(np.conjugate(np.triu(H_onsite, k=1)))
        
        return H_onsite


    def get_H_1st_nn_1_Sb(self, k):
        # Initiate zero matrix #
        H_1st_nn_1 = np.zeros((self.n_orbitals, self.n_orbitals), dtype=complex)
        
        ### Fill matrix with valuself.Es ###
        
        # Row 0
        H_1st_nn_1[0, 0] = self.get_g0_1(k) * self.V1_ss_sigma_Sb
        H_1st_nn_1[0, 2] = self.get_g1_1(k) * self.V1_sp_sigma_Sb
        H_1st_nn_1[0, 3] = self.get_g2_1(k) * self.V1_sp_sigma_Sb
        H_1st_nn_1[0, 4] = self.get_g3_1(k) * self.V1_sp_sigma_Sb
        
        # Row 1
        H_1st_nn_1[1, 1] = self.get_g0_1(k) * self.V1_ss_sigma_Sb
        H_1st_nn_1[1, 5] = self.get_g1_1(k) * self.V1_sp_sigma_Sb
        H_1st_nn_1[1, 6] = self.get_g2_1(k) * self.V1_sp_sigma_Sb
        H_1st_nn_1[1, 7] = self.get_g3_1(k) * self.V1_sp_sigma_Sb
        
        # Row 2
        H_1st_nn_1[2, 0] = -H_1st_nn_1[0, 2]
        H_1st_nn_1[2, 2] = self.get_g4_1(k) * self.V1_pp_sigma_Sb + self.get_g5_1(k) * self.V1_pp_pi_Sb
        H_1st_nn_1[2, 3] = self.get_g12_1(k) * (self.V1_pp_sigma_Sb - self.V1_pp_pi_Sb)
        H_1st_nn_1[2, 4] = self.get_g6_1(k) * (self.V1_pp_sigma_Sb - self.V1_pp_pi_Sb)
        
        # Row 3
        H_1st_nn_1[3, 0] = -H_1st_nn_1[0, 3]
        H_1st_nn_1[3, 2] = H_1st_nn_1[2, 3]
        H_1st_nn_1[3, 3] = self.get_g7_1(k) * self.V1_pp_sigma_Sb + self.get_g8_1(k) * self.V1_pp_pi_Sb
        H_1st_nn_1[3, 4] = self.get_g11_1(k) * (self.V1_pp_sigma_Sb - self.V1_pp_pi_Sb)
        
        # Row 4
        H_1st_nn_1[4, 0] = -H_1st_nn_1[0, 4]
        H_1st_nn_1[4, 2] = H_1st_nn_1[2, 4]
        H_1st_nn_1[4, 3] = H_1st_nn_1[3, 4]
        H_1st_nn_1[4, 4] = self.get_g9_1(k) * self.V1_pp_sigma_Sb + self.get_g10_1(k) * self.V1_pp_pi_Sb
        
        # Row 5
        H_1st_nn_1[5, 1] = -H_1st_nn_1[1, 5]
        H_1st_nn_1[5, 5] = H_1st_nn_1[2, 2]
        H_1st_nn_1[5, 6] = H_1st_nn_1[2, 3]
        H_1st_nn_1[5, 7] = H_1st_nn_1[2, 4]
        
        # Row 6
        H_1st_nn_1[6, 1] = -H_1st_nn_1[1, 6]
        H_1st_nn_1[6, 5] = H_1st_nn_1[3, 2]
        H_1st_nn_1[6, 6] = H_1st_nn_1[3, 3]
        H_1st_nn_1[6, 7] = H_1st_nn_1[3, 4]
        
        # Row 7
        H_1st_nn_1[7, 1] = -H_1st_nn_1[1, 7]
        H_1st_nn_1[7, 5] = H_1st_nn_1[4, 2]
        H_1st_nn_1[7, 6] = H_1st_nn_1[4, 3]
        H_1st_nn_1[7, 7] = H_1st_nn_1[4, 4]
        
        return H_1st_nn_1


    def get_H_1st_nn_2_Sb(self, k):
        # Initiate zero matrix #
        H_1st_nn_2 = np.zeros((self.n_orbitals, self.n_orbitals), dtype=complex)
        
        ### Fill matrix with valuself.Es ###
        
        # Row 0
        H_1st_nn_2[0, 0] = self.get_g0_2(k) * self.V1_ss_sigma_Sb
        H_1st_nn_2[0, 2] = self.get_g1_2(k) * self.V1_sp_sigma_Sb
        H_1st_nn_2[0, 3] = self.get_g2_2(k) * self.V1_sp_sigma_Sb
        H_1st_nn_2[0, 4] = self.get_g3_2(k) * self.V1_sp_sigma_Sb
        
        # Row 1
        H_1st_nn_2[1, 1] = self.get_g0_2(k) * self.V1_ss_sigma_Sb
        H_1st_nn_2[1, 5] = self.get_g1_2(k) * self.V1_sp_sigma_Sb
        H_1st_nn_2[1, 6] = self.get_g2_2(k) * self.V1_sp_sigma_Sb
        H_1st_nn_2[1, 7] = self.get_g3_2(k) * self.V1_sp_sigma_Sb
        
        # Row 2
        H_1st_nn_2[2, 0] = -H_1st_nn_2[0, 2]
        H_1st_nn_2[2, 2] = self.get_g4_2(k) * self.V1_pp_sigma_Sb + self.get_g5_2(k) * self.V1_pp_pi_Sb
        H_1st_nn_2[2, 3] = self.get_g12_2(k) * (self.V1_pp_sigma_Sb - self.V1_pp_pi_Sb)
        H_1st_nn_2[2, 4] = self.get_g6_2(k) * (self.V1_pp_sigma_Sb - self.V1_pp_pi_Sb)
        
        # Row 3
        H_1st_nn_2[3, 0] = -H_1st_nn_2[0, 3]
        H_1st_nn_2[3, 2] = H_1st_nn_2[2, 3]
        H_1st_nn_2[3, 3] = self.get_g7_2(k) * self.V1_pp_sigma_Sb + self.get_g8_2(k) * self.V1_pp_pi_Sb
        H_1st_nn_2[3, 4] = self.get_g11_2(k) * (self.V1_pp_sigma_Sb - self.V1_pp_pi_Sb)
        
        # Row 4
        H_1st_nn_2[4, 0] = -H_1st_nn_2[0, 4]
        H_1st_nn_2[4, 2] = H_1st_nn_2[2, 4]
        H_1st_nn_2[4, 3] = H_1st_nn_2[3, 4]
        H_1st_nn_2[4, 4] = self.get_g9_2(k) * self.V1_pp_sigma_Sb + self.get_g10_2(k) * self.V1_pp_pi_Sb
        
        # Row 5
        H_1st_nn_2[5, 1] = -H_1st_nn_2[1, 5]
        H_1st_nn_2[5, 5] = H_1st_nn_2[2, 2]
        H_1st_nn_2[5, 6] = H_1st_nn_2[2, 3]
        H_1st_nn_2[5, 7] = H_1st_nn_2[2, 4]
        
        # Row 6
        H_1st_nn_2[6, 1] = -H_1st_nn_2[1, 6]
        H_1st_nn_2[6, 5] = H_1st_nn_2[3, 2]
        H_1st_nn_2[6, 6] = H_1st_nn_2[3, 3]
        H_1st_nn_2[6, 7] = H_1st_nn_2[3, 4]
        
        # Row 7
        H_1st_nn_2[7, 1] = -H_1st_nn_2[1, 7]
        H_1st_nn_2[7, 5] = H_1st_nn_2[4, 2]
        H_1st_nn_2[7, 6] = H_1st_nn_2[4, 3]
        H_1st_nn_2[7, 7] = H_1st_nn_2[4, 4]
        
        return H_1st_nn_2


    def get_H_1st_nn_3_Sb(self, k):
        # Initiate zero matrix #
        H_1st_nn_3 = np.zeros((self.n_orbitals, self.n_orbitals), dtype=complex)
        
        ### Fill matrix with valuself.Es ###
        
        # Row 0
        H_1st_nn_3[0, 0] = self.get_g0_3(k) * self.V1_ss_sigma_Sb
        H_1st_nn_3[0, 3] = self.get_g2_3(k) * self.V1_sp_sigma_Sb
        H_1st_nn_3[0, 4] = self.get_g3_3(k) * self.V1_sp_sigma_Sb
        
        # Row 1
        H_1st_nn_3[1, 1] = self.get_g0_3(k) * self.V1_ss_sigma_Sb
        H_1st_nn_3[1, 6] = self.get_g2_3(k) * self.V1_sp_sigma_Sb
        H_1st_nn_3[1, 7] = self.get_g3_3(k) * self.V1_sp_sigma_Sb
        
        # Row 2
        H_1st_nn_3[2, 0] = -H_1st_nn_3[0, 2]
        H_1st_nn_3[2, 2] = self.get_g5_3(k) * self.V1_pp_pi_Sb
        
        # Row 3
        H_1st_nn_3[3, 0] = -H_1st_nn_3[0, 3]
        H_1st_nn_3[3, 2] = H_1st_nn_3[2, 3]
        H_1st_nn_3[3, 3] = self.get_g7_3(k) * self.V1_pp_sigma_Sb + self.get_g8_3(k) * self.V1_pp_pi_Sb
        H_1st_nn_3[3, 4] = self.get_g11_3(k) * (self.V1_pp_sigma_Sb - self.V1_pp_pi_Sb)
        
        # Row 4
        H_1st_nn_3[4, 0] = -H_1st_nn_3[0, 4]
        H_1st_nn_3[4, 2] = H_1st_nn_3[2, 4]
        H_1st_nn_3[4, 3] = H_1st_nn_3[3, 4]
        H_1st_nn_3[4, 4] = self.get_g9_3(k) * self.V1_pp_sigma_Sb + self.get_g10_3(k) * self.V1_pp_pi_Sb
        
        # Row 5
        H_1st_nn_3[5, 1] = -H_1st_nn_3[1, 5]
        H_1st_nn_3[5, 5] = H_1st_nn_3[2, 2]
        H_1st_nn_3[5, 6] = H_1st_nn_3[2, 3]
        H_1st_nn_3[5, 7] = H_1st_nn_3[2, 4]
        
        # Row 6
        H_1st_nn_3[6, 1] = -H_1st_nn_3[1, 6]
        H_1st_nn_3[6, 5] = H_1st_nn_3[3, 2]
        H_1st_nn_3[6, 6] = H_1st_nn_3[3, 3]
        H_1st_nn_3[6, 7] = H_1st_nn_3[3, 4]
        
        # Row 7
        H_1st_nn_3[7, 1] = -H_1st_nn_3[1, 7]
        H_1st_nn_3[7, 5] = H_1st_nn_3[4, 2]
        H_1st_nn_3[7, 6] = H_1st_nn_3[4, 3]
        H_1st_nn_3[7, 7] = H_1st_nn_3[4, 4]
        
        return H_1st_nn_3


    def get_H_2nd_nn_1_Sb(self, k):
        # Initiate zero matrix #
        H_2nd_nn_1 = np.zeros((self.n_orbitals, self.n_orbitals), dtype=complex)
        
        ### Fill matrix with valuself.Es ###
        
        # Row 0
        H_2nd_nn_1[0, 0] = self.get_g13_1(k) * self.V2_ss_sigma_Sb
        H_2nd_nn_1[0, 2] = self.get_g14_1(k) * self.V2_sp_sigma_Sb
        H_2nd_nn_1[0, 3] = self.get_g15_1(k) * self.V2_sp_sigma_Sb
        H_2nd_nn_1[0, 4] = self.get_g16_1(k) * self.V2_sp_sigma_Sb
        
        # Row 1
        H_2nd_nn_1[1, 1] = self.get_g13_1(k) * self.V2_ss_sigma_Sb
        H_2nd_nn_1[1, 5] = self.get_g14_1(k) * self.V2_sp_sigma_Sb
        H_2nd_nn_1[1, 6] = self.get_g15_1(k) * self.V2_sp_sigma_Sb
        H_2nd_nn_1[1, 7] = self.get_g16_1(k) * self.V2_sp_sigma_Sb
        
        # Row 2
        H_2nd_nn_1[2, 0] = -H_2nd_nn_1[0, 2]
        H_2nd_nn_1[2, 2] = self.get_g17_1(k) * self.V2_pp_sigma_Sb + self.get_g18_1(k) * self.V2_pp_pi_Sb
        H_2nd_nn_1[2, 3] = self.get_g25_1(k) * (self.V2_pp_sigma_Sb - self.V2_pp_pi_Sb)
        H_2nd_nn_1[2, 4] = self.get_g19_1(k) * (self.V2_pp_sigma_Sb - self.V2_pp_pi_Sb)
        
        # Row 3
        H_2nd_nn_1[3, 0] = -H_2nd_nn_1[0, 3]
        H_2nd_nn_1[3, 2] = H_2nd_nn_1[2, 3]
        H_2nd_nn_1[3, 3] = self.get_g20_1(k) * self.V2_pp_sigma_Sb + self.get_g21_1(k) * self.V2_pp_pi_Sb
        H_2nd_nn_1[3, 4] = self.get_g24_1(k) * (self.V2_pp_sigma_Sb - self.V2_pp_pi_Sb)
        
        # Row 4
        H_2nd_nn_1[4, 0] = -H_2nd_nn_1[0, 4]
        H_2nd_nn_1[4, 2] = H_2nd_nn_1[2, 4]
        H_2nd_nn_1[4, 3] = H_2nd_nn_1[3, 4]
        H_2nd_nn_1[4, 4] = self.get_g22_1(k) * self.V2_pp_sigma_Sb + self.get_g23_1(k) * self.V2_pp_pi_Sb
        
        # Row 5
        H_2nd_nn_1[5, 1] = -H_2nd_nn_1[1, 5]
        H_2nd_nn_1[5, 5] = H_2nd_nn_1[2, 2]
        H_2nd_nn_1[5, 6] = H_2nd_nn_1[2, 3]
        H_2nd_nn_1[5, 7] = H_2nd_nn_1[2, 4]
        
        # Row 6
        H_2nd_nn_1[6, 1] = -H_2nd_nn_1[1, 6]
        H_2nd_nn_1[6, 5] = H_2nd_nn_1[3, 2]
        H_2nd_nn_1[6, 6] = H_2nd_nn_1[3, 3]
        H_2nd_nn_1[6, 7] = H_2nd_nn_1[3, 4]
        
        # Row 7
        H_2nd_nn_1[7, 1] = -H_2nd_nn_1[1, 7]
        H_2nd_nn_1[7, 5] = H_2nd_nn_1[4, 2]
        H_2nd_nn_1[7, 6] = H_2nd_nn_1[4, 3]
        H_2nd_nn_1[7, 7] = H_2nd_nn_1[4, 4]
        
        return H_2nd_nn_1


    def get_H_2nd_nn_2_Sb(self, k):
        # Initiate zero matrix #
        H_2nd_nn_2 = np.zeros((self.n_orbitals, self.n_orbitals), dtype=complex)
        
        ### Fill matrix with valuself.Es ###
        
        # Row 0
        H_2nd_nn_2[0, 0] = self.get_g13_2(k) * self.V2_ss_sigma_Sb
        H_2nd_nn_2[0, 2] = self.get_g14_2(k) * self.V2_sp_sigma_Sb
        H_2nd_nn_2[0, 3] = self.get_g15_2(k) * self.V2_sp_sigma_Sb
        H_2nd_nn_2[0, 4] = self.get_g16_2(k) * self.V2_sp_sigma_Sb
        
        # Row 1
        H_2nd_nn_2[1, 1] = self.get_g13_2(k) * self.V2_ss_sigma_Sb
        H_2nd_nn_2[1, 5] = self.get_g14_2(k) * self.V2_sp_sigma_Sb
        H_2nd_nn_2[1, 6] = self.get_g15_2(k) * self.V2_sp_sigma_Sb
        H_2nd_nn_2[1, 7] = self.get_g16_2(k) * self.V2_sp_sigma_Sb
        
        # Row 2
        H_2nd_nn_2[2, 0] = -H_2nd_nn_2[0, 2]
        H_2nd_nn_2[2, 2] = self.get_g17_2(k) * self.V2_pp_sigma_Sb + self.get_g18_2(k) * self.V2_pp_pi_Sb
        H_2nd_nn_2[2, 3] = self.get_g25_2(k) * (self.V2_pp_sigma_Sb - self.V2_pp_pi_Sb)
        H_2nd_nn_2[2, 4] = self.get_g19_2(k) * (self.V2_pp_sigma_Sb - self.V2_pp_pi_Sb)
        
        # Row 3
        H_2nd_nn_2[3, 0] = -H_2nd_nn_2[0, 3]
        H_2nd_nn_2[3, 2] = H_2nd_nn_2[2, 3]
        H_2nd_nn_2[3, 3] = self.get_g20_2(k) * self.V2_pp_sigma_Sb + self.get_g21_2(k) * self.V2_pp_pi_Sb
        H_2nd_nn_2[3, 4] = self.get_g24_2(k) * (self.V2_pp_sigma_Sb - self.V2_pp_pi_Sb)
        
        # Row 4
        H_2nd_nn_2[4, 0] = -H_2nd_nn_2[0, 4]
        H_2nd_nn_2[4, 2] = H_2nd_nn_2[2, 4]
        H_2nd_nn_2[4, 3] = H_2nd_nn_2[3, 4]
        H_2nd_nn_2[4, 4] = self.get_g22_2(k) * self.V2_pp_sigma_Sb + self.get_g23_2(k) * self.V2_pp_pi_Sb
        
        # Row 5
        H_2nd_nn_2[5, 1] = -H_2nd_nn_2[1, 5]
        H_2nd_nn_2[5, 5] = H_2nd_nn_2[2, 2]
        H_2nd_nn_2[5, 6] = H_2nd_nn_2[2, 3]
        H_2nd_nn_2[5, 7] = H_2nd_nn_2[2, 4]
        
        # Row 6
        H_2nd_nn_2[6, 1] = -H_2nd_nn_2[1, 6]
        H_2nd_nn_2[6, 5] = H_2nd_nn_2[3, 2]
        H_2nd_nn_2[6, 6] = H_2nd_nn_2[3, 3]
        H_2nd_nn_2[6, 7] = H_2nd_nn_2[3, 4]
        
        # Row 7
        H_2nd_nn_2[7, 1] = -H_2nd_nn_2[1, 7]
        H_2nd_nn_2[7, 5] = H_2nd_nn_2[4, 2]
        H_2nd_nn_2[7, 6] = H_2nd_nn_2[4, 3]
        H_2nd_nn_2[7, 7] = H_2nd_nn_2[4, 4]
        
        return H_2nd_nn_2


    def get_H_2nd_nn_3_Sb(self, k):
        # Initiate zero matrix #
        H_2nd_nn_3 = np.zeros((self.n_orbitals, self.n_orbitals), dtype=complex)
        
        ### Fill matrix with valuself.Es ###
        
        # Row 0
        H_2nd_nn_3[0, 0] = self.get_g13_3(k) * self.V2_ss_sigma_Sb
        H_2nd_nn_3[0, 3] = self.get_g15_3(k) * self.V2_sp_sigma_Sb
        H_2nd_nn_3[0, 4] = self.get_g16_3(k) * self.V2_sp_sigma_Sb
        
        # Row 1
        H_2nd_nn_3[1, 1] = self.get_g13_3(k) * self.V2_ss_sigma_Sb
        H_2nd_nn_3[1, 6] = self.get_g15_3(k) * self.V2_sp_sigma_Sb
        H_2nd_nn_3[1, 7] = self.get_g16_3(k) * self.V2_sp_sigma_Sb
        
        # Row 2
        H_2nd_nn_3[2, 0] = -H_2nd_nn_3[0, 2]
        H_2nd_nn_3[2, 2] = self.get_g18_3(k) * self.V2_pp_pi_Sb
        
        # Row 3
        H_2nd_nn_3[3, 0] = -H_2nd_nn_3[0, 3]
        H_2nd_nn_3[3, 2] = H_2nd_nn_3[2, 3]
        H_2nd_nn_3[3, 3] = self.get_g20_3(k) * self.V2_pp_sigma_Sb + self.get_g21_3(k) * self.V2_pp_pi_Sb
        H_2nd_nn_3[3, 4] = self.get_g24_3(k) * (self.V2_pp_sigma_Sb - self.V2_pp_pi_Sb)
        
        # Row 4
        H_2nd_nn_3[4, 0] = -H_2nd_nn_3[0, 4]
        H_2nd_nn_3[4, 2] = H_2nd_nn_3[2, 4]
        H_2nd_nn_3[4, 3] = H_2nd_nn_3[3, 4]
        H_2nd_nn_3[4, 4] = self.get_g22_3(k) * self.V2_pp_sigma_Sb + self.get_g23_3(k) * self.V2_pp_pi_Sb
        
        # Row 5
        H_2nd_nn_3[5, 1] = -H_2nd_nn_3[1, 5]
        H_2nd_nn_3[5, 5] = H_2nd_nn_3[2, 2]
        H_2nd_nn_3[5, 6] = H_2nd_nn_3[2, 3]
        H_2nd_nn_3[5, 7] = H_2nd_nn_3[2, 4]
        
        # Row 6
        H_2nd_nn_3[6, 1] = -H_2nd_nn_3[1, 6]
        H_2nd_nn_3[6, 5] = H_2nd_nn_3[3, 2]
        H_2nd_nn_3[6, 6] = H_2nd_nn_3[3, 3]
        H_2nd_nn_3[6, 7] = H_2nd_nn_3[3, 4]
        
        # Row 7
        H_2nd_nn_3[7, 1] = -H_2nd_nn_3[1, 7]
        H_2nd_nn_3[7, 5] = H_2nd_nn_3[4, 2]
        H_2nd_nn_3[7, 6] = H_2nd_nn_3[4, 3]
        H_2nd_nn_3[7, 7] = H_2nd_nn_3[4, 4]
        
        return H_2nd_nn_3


    def get_H_3rd_nn_1_Sb(self, k):
        # Initiate zero matrix #
        H_3rd_nn_1 = np.zeros((self.n_orbitals, self.n_orbitals), dtype=complex)
        
        ### Fill upper triangular part of matrix with valuself.Es ###
        
        # Row 0
        H_3rd_nn_1[0, 0] = self.get_g26_1(k) * self.V3_ss_sigma_Sb
        H_3rd_nn_1[0, 2] = self.get_g27_1(k) * self.V3_sp_sigma_Sb
        H_3rd_nn_1[0, 3] = self.get_g28_1(k) * self.V3_sp_sigma_Sb
        
        # Row 1
        H_3rd_nn_1[1, 1] = self.get_g26_1(k) * self.V3_ss_sigma_Sb
        H_3rd_nn_1[1, 5] = self.get_g27_1(k) * self.V3_sp_sigma_Sb
        H_3rd_nn_1[1, 6] = self.get_g28_1(k) * self.V3_sp_sigma_Sb
        
        # Row 2
        H_3rd_nn_1[2, 2] = self.get_g29_1(k) * self.V3_pp_sigma_Sb + self.get_g30_1(k) * self.V3_pp_pi_Sb
        H_3rd_nn_1[2, 3] = self.get_g31_1(k) * (self.V3_pp_sigma_Sb - self.V3_pp_pi_Sb)
        
        # Row 3
        H_3rd_nn_1[3, 3] = self.get_g30_1(k) * self.V3_pp_sigma_Sb + self.get_g29_1(k) * self.V3_pp_pi_Sb
        
        # Row 4
        H_3rd_nn_1[4, 4] = self.get_g26_1(k) * self.V3_pp_pi_Sb
        
        # Row 5
        H_3rd_nn_1[5, 5] = self.get_g29_1(k) * self.V3_pp_sigma_Sb + self.get_g30_1(k) * self.V3_pp_pi_Sb
        H_3rd_nn_1[5, 6] = self.get_g31_1(k) * (self.V3_pp_sigma_Sb - self.V3_pp_pi_Sb)
        
        # Row 6
        H_3rd_nn_1[6, 6] = self.get_g30_1(k) * self.V3_pp_sigma_Sb + self.get_g29_1(k) * self.V3_pp_pi_Sb
        
        # Row 7
        H_3rd_nn_1[7, 7] = self.get_g26_1(k) * self.V3_pp_pi_Sb
        
        # Add the hermitian conjugate of the upper triangular to get hermitian matrix #
        H_3rd_nn_1 += np.transpose(np.triu(H_3rd_nn_1, k=1))
        
        return H_3rd_nn_1


    def get_H_3rd_nn_2_Sb(self, k):
        # Initiate zero matrix #
        H_3rd_nn_2 = np.zeros((self.n_orbitals, self.n_orbitals), dtype=complex)
        
        ### Fill upper triangular part of matrix with valuself.Es ###
        
        # Row 0
        H_3rd_nn_2[0, 0] = self.get_g26_2(k) * self.V3_ss_sigma_Sb
        H_3rd_nn_2[0, 2] = self.get_g27_2(k) * self.V3_sp_sigma_Sb
        H_3rd_nn_2[0, 3] = self.get_g28_2(k) * self.V3_sp_sigma_Sb
        
        # Row 1
        H_3rd_nn_2[1, 1] = self.get_g26_2(k) * self.V3_ss_sigma_Sb
        H_3rd_nn_2[1, 5] = self.get_g27_2(k) * self.V3_sp_sigma_Sb
        H_3rd_nn_2[1, 6] = self.get_g28_2(k) * self.V3_sp_sigma_Sb
        
        # Row 2
        H_3rd_nn_2[2, 2] = self.get_g29_2(k) * self.V3_pp_sigma_Sb + self.get_g30_2(k) * self.V3_pp_pi_Sb
        H_3rd_nn_2[2, 3] = self.get_g31_2(k) * (self.V3_pp_sigma_Sb - self.V3_pp_pi_Sb)
        
        # Row 3
        H_3rd_nn_2[3, 3] = self.get_g30_2(k) * self.V3_pp_sigma_Sb + self.get_g29_2(k) * self.V3_pp_pi_Sb
        
        # Row 4
        H_3rd_nn_2[4, 4] = self.get_g26_2(k) * self.V3_pp_pi_Sb
        
        # Row 5
        H_3rd_nn_2[5, 5] = self.get_g29_2(k) * self.V3_pp_sigma_Sb + self.get_g30_2(k) * self.V3_pp_pi_Sb
        H_3rd_nn_2[5, 6] = self.get_g31_2(k) * (self.V3_pp_sigma_Sb - self.V3_pp_pi_Sb)
        
        # Row 6
        H_3rd_nn_2[6, 6] = self.get_g30_2(k) * self.V3_pp_sigma_Sb + self.get_g29_2(k) * self.V3_pp_pi_Sb
        
        # Row 7
        H_3rd_nn_2[7, 7] = self.get_g26_2(k) * self.V3_pp_pi_Sb
        
        # Add the hermitian conjugate of the upper triangular to get hermition matrix #
        H_3rd_nn_2 += np.transpose(np.triu(H_3rd_nn_2, k=1))
        
        return H_3rd_nn_2


    def get_H_3rd_nn_3_Sb(self, k):
        # Initiate zero matrix #
        H_3rd_nn_3 = np.zeros((self.n_orbitals, self.n_orbitals), dtype=complex)
        
        ### Fill upper triangular part of matrix with valuself.Es ###
        
        # Row 0
        H_3rd_nn_3[0, 0] = self.get_g26_3(k) * self.V3_ss_sigma_Sb
        H_3rd_nn_3[0, 2] = self.get_g27_3(k) * self.V3_sp_sigma_Sb
        
        # Row 1
        H_3rd_nn_3[1, 1] = self.get_g26_3(k) * self.V3_ss_sigma_Sb
        H_3rd_nn_3[1, 5] = self.get_g27_3(k) * self.V3_sp_sigma_Sb
        
        # Row 2
        H_3rd_nn_3[2, 2] = self.get_g29_3(k) * self.V3_pp_sigma_Sb
        
        # Row 3
        H_3rd_nn_3[3, 3] = self.get_g29_3(k) * self.V3_pp_pi_Sb
        
        # Row 4
        H_3rd_nn_3[4, 4] = self.get_g26_3(k) * self.V3_pp_pi_Sb
        
        # Row 5
        H_3rd_nn_3[5, 5] = self.get_g29_3(k) * self.V3_pp_sigma_Sb
        
        # Row 6
        H_3rd_nn_3[6, 6] = self.get_g29_3(k) * self.V3_pp_pi_Sb
        
        # Row 7
        H_3rd_nn_3[7, 7] = self.get_g26_3(k) * self.V3_pp_pi_Sb
        
        # Add the hermitian conjugate of the upper triangular to get hermition matrix #
        H_3rd_nn_3 += np.transpose(np.triu(H_3rd_nn_3, k=1))
        
        return H_3rd_nn_3


    def get_H_all_nn_Sb(self):
        return [
            [self.get_H_1st_nn_1_Sb, self.get_H_1st_nn_2_Sb, self.get_H_1st_nn_3_Sb],
            [self.get_H_2nd_nn_1_Sb, self.get_H_2nd_nn_2_Sb, self.get_H_2nd_nn_3_Sb],
            [self.get_H_3rd_nn_1_Sb, self.get_H_3rd_nn_2_Sb, self.get_H_3rd_nn_3_Sb]
            ]


    def get_H_1st_nn_1_Bi_Sb(self, k):
        # Initiate zero matrix #
        H_1st_nn_1 = np.zeros((self.n_orbitals, self.n_orbitals), dtype=complex)
        
        ### Fill matrix with valuself.Es ###
        
        # Row 0
        H_1st_nn_1[0, 0] = self.get_g0_1(k) * self.V1_ss_sigma_Bi_Sb
        H_1st_nn_1[0, 2] = self.get_g1_1(k) * self.V1_sp_sigma_Bi_Sb
        H_1st_nn_1[0, 3] = self.get_g2_1(k) * self.V1_sp_sigma_Bi_Sb
        H_1st_nn_1[0, 4] = self.get_g3_1(k) * self.V1_sp_sigma_Bi_Sb
        
        # Row 1
        H_1st_nn_1[1, 1] = self.get_g0_1(k) * self.V1_ss_sigma_Bi_Sb
        H_1st_nn_1[1, 5] = self.get_g1_1(k) * self.V1_sp_sigma_Bi_Sb
        H_1st_nn_1[1, 6] = self.get_g2_1(k) * self.V1_sp_sigma_Bi_Sb
        H_1st_nn_1[1, 7] = self.get_g3_1(k) * self.V1_sp_sigma_Bi_Sb
        
        # Row 2
        H_1st_nn_1[2, 0] = -H_1st_nn_1[0, 2]
        H_1st_nn_1[2, 2] = self.get_g4_1(k) * self.V1_pp_sigma_Bi_Sb + self.get_g5_1(k) * self.V1_pp_pi_Bi_Sb
        H_1st_nn_1[2, 3] = self.get_g12_1(k) * (self.V1_pp_sigma_Bi_Sb - self.V1_pp_pi_Bi_Sb)
        H_1st_nn_1[2, 4] = self.get_g6_1(k) * (self.V1_pp_sigma_Bi_Sb - self.V1_pp_pi_Bi_Sb)
        
        # Row 3
        H_1st_nn_1[3, 0] = -H_1st_nn_1[0, 3]
        H_1st_nn_1[3, 2] = H_1st_nn_1[2, 3]
        H_1st_nn_1[3, 3] = self.get_g7_1(k) * self.V1_pp_sigma_Bi_Sb + self.get_g8_1(k) * self.V1_pp_pi_Bi_Sb
        H_1st_nn_1[3, 4] = self.get_g11_1(k) * (self.V1_pp_sigma_Bi_Sb - self.V1_pp_pi_Bi_Sb)
        
        # Row 4
        H_1st_nn_1[4, 0] = -H_1st_nn_1[0, 4]
        H_1st_nn_1[4, 2] = H_1st_nn_1[2, 4]
        H_1st_nn_1[4, 3] = H_1st_nn_1[3, 4]
        H_1st_nn_1[4, 4] = self.get_g9_1(k) * self.V1_pp_sigma_Bi_Sb + self.get_g10_1(k) * self.V1_pp_pi_Bi_Sb
        
        # Row 5
        H_1st_nn_1[5, 1] = -H_1st_nn_1[1, 5]
        H_1st_nn_1[5, 5] = H_1st_nn_1[2, 2]
        H_1st_nn_1[5, 6] = H_1st_nn_1[2, 3]
        H_1st_nn_1[5, 7] = H_1st_nn_1[2, 4]
        
        # Row 6
        H_1st_nn_1[6, 1] = -H_1st_nn_1[1, 6]
        H_1st_nn_1[6, 5] = H_1st_nn_1[3, 2]
        H_1st_nn_1[6, 6] = H_1st_nn_1[3, 3]
        H_1st_nn_1[6, 7] = H_1st_nn_1[3, 4]
        
        # Row 7
        H_1st_nn_1[7, 1] = -H_1st_nn_1[1, 7]
        H_1st_nn_1[7, 5] = H_1st_nn_1[4, 2]
        H_1st_nn_1[7, 6] = H_1st_nn_1[4, 3]
        H_1st_nn_1[7, 7] = H_1st_nn_1[4, 4]
        
        return H_1st_nn_1


    def get_H_1st_nn_2_Bi_Sb(self, k):
        # Initiate zero matrix #
        H_1st_nn_2 = np.zeros((self.n_orbitals, self.n_orbitals), dtype=complex)
        
        ### Fill matrix with valuself.Es ###
        
        # Row 0
        H_1st_nn_2[0, 0] = self.get_g0_2(k) * self.V1_ss_sigma_Bi_Sb
        H_1st_nn_2[0, 2] = self.get_g1_2(k) * self.V1_sp_sigma_Bi_Sb
        H_1st_nn_2[0, 3] = self.get_g2_2(k) * self.V1_sp_sigma_Bi_Sb
        H_1st_nn_2[0, 4] = self.get_g3_2(k) * self.V1_sp_sigma_Bi_Sb
        
        # Row 1
        H_1st_nn_2[1, 1] = self.get_g0_2(k) * self.V1_ss_sigma_Bi_Sb
        H_1st_nn_2[1, 5] = self.get_g1_2(k) * self.V1_sp_sigma_Bi_Sb
        H_1st_nn_2[1, 6] = self.get_g2_2(k) * self.V1_sp_sigma_Bi_Sb
        H_1st_nn_2[1, 7] = self.get_g3_2(k) * self.V1_sp_sigma_Bi_Sb
        
        # Row 2
        H_1st_nn_2[2, 0] = -H_1st_nn_2[0, 2]
        H_1st_nn_2[2, 2] = self.get_g4_2(k) * self.V1_pp_sigma_Bi_Sb + self.get_g5_2(k) * self.V1_pp_pi_Bi_Sb
        H_1st_nn_2[2, 3] = self.get_g12_2(k) * (self.V1_pp_sigma_Bi_Sb - self.V1_pp_pi_Bi_Sb)
        H_1st_nn_2[2, 4] = self.get_g6_2(k) * (self.V1_pp_sigma_Bi_Sb - self.V1_pp_pi_Bi_Sb)
        
        # Row 3
        H_1st_nn_2[3, 0] = -H_1st_nn_2[0, 3]
        H_1st_nn_2[3, 2] = H_1st_nn_2[2, 3]
        H_1st_nn_2[3, 3] = self.get_g7_2(k) * self.V1_pp_sigma_Bi_Sb + self.get_g8_2(k) * self.V1_pp_pi_Bi_Sb
        H_1st_nn_2[3, 4] = self.get_g11_2(k) * (self.V1_pp_sigma_Bi_Sb - self.V1_pp_pi_Bi_Sb)
        
        # Row 4
        H_1st_nn_2[4, 0] = -H_1st_nn_2[0, 4]
        H_1st_nn_2[4, 2] = H_1st_nn_2[2, 4]
        H_1st_nn_2[4, 3] = H_1st_nn_2[3, 4]
        H_1st_nn_2[4, 4] = self.get_g9_2(k) * self.V1_pp_sigma_Bi_Sb + self.get_g10_2(k) * self.V1_pp_pi_Bi_Sb
        
        # Row 5
        H_1st_nn_2[5, 1] = -H_1st_nn_2[1, 5]
        H_1st_nn_2[5, 5] = H_1st_nn_2[2, 2]
        H_1st_nn_2[5, 6] = H_1st_nn_2[2, 3]
        H_1st_nn_2[5, 7] = H_1st_nn_2[2, 4]
        
        # Row 6
        H_1st_nn_2[6, 1] = -H_1st_nn_2[1, 6]
        H_1st_nn_2[6, 5] = H_1st_nn_2[3, 2]
        H_1st_nn_2[6, 6] = H_1st_nn_2[3, 3]
        H_1st_nn_2[6, 7] = H_1st_nn_2[3, 4]
        
        # Row 7
        H_1st_nn_2[7, 1] = -H_1st_nn_2[1, 7]
        H_1st_nn_2[7, 5] = H_1st_nn_2[4, 2]
        H_1st_nn_2[7, 6] = H_1st_nn_2[4, 3]
        H_1st_nn_2[7, 7] = H_1st_nn_2[4, 4]
        
        return H_1st_nn_2


    def get_H_1st_nn_3_Bi_Sb(self, k):
        # Initiate zero matrix #
        H_1st_nn_3 = np.zeros((self.n_orbitals, self.n_orbitals), dtype=complex)
        
        ### Fill matrix with valuself.Es ###
        
        # Row 0
        H_1st_nn_3[0, 0] = self.get_g0_3(k) * self.V1_ss_sigma_Bi_Sb
        H_1st_nn_3[0, 3] = self.get_g2_3(k) * self.V1_sp_sigma_Bi_Sb
        H_1st_nn_3[0, 4] = self.get_g3_3(k) * self.V1_sp_sigma_Bi_Sb
        
        # Row 1
        H_1st_nn_3[1, 1] = self.get_g0_3(k) * self.V1_ss_sigma_Bi_Sb
        H_1st_nn_3[1, 6] = self.get_g2_3(k) * self.V1_sp_sigma_Bi_Sb
        H_1st_nn_3[1, 7] = self.get_g3_3(k) * self.V1_sp_sigma_Bi_Sb
        
        # Row 2
        H_1st_nn_3[2, 0] = -H_1st_nn_3[0, 2]
        H_1st_nn_3[2, 2] = self.get_g5_3(k) * self.V1_pp_pi_Bi_Sb
        
        # Row 3
        H_1st_nn_3[3, 0] = -H_1st_nn_3[0, 3]
        H_1st_nn_3[3, 2] = H_1st_nn_3[2, 3]
        H_1st_nn_3[3, 3] = self.get_g7_3(k) * self.V1_pp_sigma_Bi_Sb + self.get_g8_3(k) * self.V1_pp_pi_Bi_Sb
        H_1st_nn_3[3, 4] = self.get_g11_3(k) * (self.V1_pp_sigma_Bi_Sb - self.V1_pp_pi_Bi_Sb)
        
        # Row 4
        H_1st_nn_3[4, 0] = -H_1st_nn_3[0, 4]
        H_1st_nn_3[4, 2] = H_1st_nn_3[2, 4]
        H_1st_nn_3[4, 3] = H_1st_nn_3[3, 4]
        H_1st_nn_3[4, 4] = self.get_g9_3(k) * self.V1_pp_sigma_Bi_Sb + self.get_g10_3(k) * self.V1_pp_pi_Bi_Sb
        
        # Row 5
        H_1st_nn_3[5, 1] = -H_1st_nn_3[1, 5]
        H_1st_nn_3[5, 5] = H_1st_nn_3[2, 2]
        H_1st_nn_3[5, 6] = H_1st_nn_3[2, 3]
        H_1st_nn_3[5, 7] = H_1st_nn_3[2, 4]
        
        # Row 6
        H_1st_nn_3[6, 1] = -H_1st_nn_3[1, 6]
        H_1st_nn_3[6, 5] = H_1st_nn_3[3, 2]
        H_1st_nn_3[6, 6] = H_1st_nn_3[3, 3]
        H_1st_nn_3[6, 7] = H_1st_nn_3[3, 4]
        
        # Row 7
        H_1st_nn_3[7, 1] = -H_1st_nn_3[1, 7]
        H_1st_nn_3[7, 5] = H_1st_nn_3[4, 2]
        H_1st_nn_3[7, 6] = H_1st_nn_3[4, 3]
        H_1st_nn_3[7, 7] = H_1st_nn_3[4, 4]
        
        return H_1st_nn_3


    def get_H_2nd_nn_1_Bi_Sb(self, k):
        # Initiate zero matrix #
        H_2nd_nn_1 = np.zeros((self.n_orbitals, self.n_orbitals), dtype=complex)
        
        ### Fill matrix with valuself.Es ###
        
        # Row 0
        H_2nd_nn_1[0, 0] = self.get_g13_1(k) * self.V2_ss_sigma_Bi_Sb
        H_2nd_nn_1[0, 2] = self.get_g14_1(k) * self.V2_sp_sigma_Bi_Sb
        H_2nd_nn_1[0, 3] = self.get_g15_1(k) * self.V2_sp_sigma_Bi_Sb
        H_2nd_nn_1[0, 4] = self.get_g16_1(k) * self.V2_sp_sigma_Bi_Sb
        
        # Row 1
        H_2nd_nn_1[1, 1] = self.get_g13_1(k) * self.V2_ss_sigma_Bi_Sb
        H_2nd_nn_1[1, 5] = self.get_g14_1(k) * self.V2_sp_sigma_Bi_Sb
        H_2nd_nn_1[1, 6] = self.get_g15_1(k) * self.V2_sp_sigma_Bi_Sb
        H_2nd_nn_1[1, 7] = self.get_g16_1(k) * self.V2_sp_sigma_Bi_Sb
        
        # Row 2
        H_2nd_nn_1[2, 0] = -H_2nd_nn_1[0, 2]
        H_2nd_nn_1[2, 2] = self.get_g17_1(k) * self.V2_pp_sigma_Bi_Sb + self.get_g18_1(k) * self.V2_pp_pi_Bi_Sb
        H_2nd_nn_1[2, 3] = self.get_g25_1(k) * (self.V2_pp_sigma_Bi_Sb - self.V2_pp_pi_Bi_Sb)
        H_2nd_nn_1[2, 4] = self.get_g19_1(k) * (self.V2_pp_sigma_Bi_Sb - self.V2_pp_pi_Bi_Sb)
        
        # Row 3
        H_2nd_nn_1[3, 0] = -H_2nd_nn_1[0, 3]
        H_2nd_nn_1[3, 2] = H_2nd_nn_1[2, 3]
        H_2nd_nn_1[3, 3] = self.get_g20_1(k) * self.V2_pp_sigma_Bi_Sb + self.get_g21_1(k) * self.V2_pp_pi_Bi_Sb
        H_2nd_nn_1[3, 4] = self.get_g24_1(k) * (self.V2_pp_sigma_Bi_Sb - self.V2_pp_pi_Bi_Sb)
        
        # Row 4
        H_2nd_nn_1[4, 0] = -H_2nd_nn_1[0, 4]
        H_2nd_nn_1[4, 2] = H_2nd_nn_1[2, 4]
        H_2nd_nn_1[4, 3] = H_2nd_nn_1[3, 4]
        H_2nd_nn_1[4, 4] = self.get_g22_1(k) * self.V2_pp_sigma_Bi_Sb + self.get_g23_1(k) * self.V2_pp_pi_Bi_Sb
        
        # Row 5
        H_2nd_nn_1[5, 1] = -H_2nd_nn_1[1, 5]
        H_2nd_nn_1[5, 5] = H_2nd_nn_1[2, 2]
        H_2nd_nn_1[5, 6] = H_2nd_nn_1[2, 3]
        H_2nd_nn_1[5, 7] = H_2nd_nn_1[2, 4]
        
        # Row 6
        H_2nd_nn_1[6, 1] = -H_2nd_nn_1[1, 6]
        H_2nd_nn_1[6, 5] = H_2nd_nn_1[3, 2]
        H_2nd_nn_1[6, 6] = H_2nd_nn_1[3, 3]
        H_2nd_nn_1[6, 7] = H_2nd_nn_1[3, 4]
        
        # Row 7
        H_2nd_nn_1[7, 1] = -H_2nd_nn_1[1, 7]
        H_2nd_nn_1[7, 5] = H_2nd_nn_1[4, 2]
        H_2nd_nn_1[7, 6] = H_2nd_nn_1[4, 3]
        H_2nd_nn_1[7, 7] = H_2nd_nn_1[4, 4]
        
        return H_2nd_nn_1


    def get_H_2nd_nn_2_Bi_Sb(self, k):
        # Initiate zero matrix #
        H_2nd_nn_2 = np.zeros((self.n_orbitals, self.n_orbitals), dtype=complex)
        
        ### Fill matrix with valuself.Es ###
        
        # Row 0
        H_2nd_nn_2[0, 0] = self.get_g13_2(k) * self.V2_ss_sigma_Bi_Sb
        H_2nd_nn_2[0, 2] = self.get_g14_2(k) * self.V2_sp_sigma_Bi_Sb
        H_2nd_nn_2[0, 3] = self.get_g15_2(k) * self.V2_sp_sigma_Bi_Sb
        H_2nd_nn_2[0, 4] = self.get_g16_2(k) * self.V2_sp_sigma_Bi_Sb
        
        # Row 1
        H_2nd_nn_2[1, 1] = self.get_g13_2(k) * self.V2_ss_sigma_Bi_Sb
        H_2nd_nn_2[1, 5] = self.get_g14_2(k) * self.V2_sp_sigma_Bi_Sb
        H_2nd_nn_2[1, 6] = self.get_g15_2(k) * self.V2_sp_sigma_Bi_Sb
        H_2nd_nn_2[1, 7] = self.get_g16_2(k) * self.V2_sp_sigma_Bi_Sb
        
        # Row 2
        H_2nd_nn_2[2, 0] = -H_2nd_nn_2[0, 2]
        H_2nd_nn_2[2, 2] = self.get_g17_2(k) * self.V2_pp_sigma_Bi_Sb + self.get_g18_2(k) * self.V2_pp_pi_Bi_Sb
        H_2nd_nn_2[2, 3] = self.get_g25_2(k) * (self.V2_pp_sigma_Bi_Sb - self.V2_pp_pi_Bi_Sb)
        H_2nd_nn_2[2, 4] = self.get_g19_2(k) * (self.V2_pp_sigma_Bi_Sb - self.V2_pp_pi_Bi_Sb)
        
        # Row 3
        H_2nd_nn_2[3, 0] = -H_2nd_nn_2[0, 3]
        H_2nd_nn_2[3, 2] = H_2nd_nn_2[2, 3]
        H_2nd_nn_2[3, 3] = self.get_g20_2(k) * self.V2_pp_sigma_Bi_Sb + self.get_g21_2(k) * self.V2_pp_pi_Bi_Sb
        H_2nd_nn_2[3, 4] = self.get_g24_2(k) * (self.V2_pp_sigma_Bi_Sb - self.V2_pp_pi_Bi_Sb)
        
        # Row 4
        H_2nd_nn_2[4, 0] = -H_2nd_nn_2[0, 4]
        H_2nd_nn_2[4, 2] = H_2nd_nn_2[2, 4]
        H_2nd_nn_2[4, 3] = H_2nd_nn_2[3, 4]
        H_2nd_nn_2[4, 4] = self.get_g22_2(k) * self.V2_pp_sigma_Bi_Sb + self.get_g23_2(k) * self.V2_pp_pi_Bi_Sb
        
        # Row 5
        H_2nd_nn_2[5, 1] = -H_2nd_nn_2[1, 5]
        H_2nd_nn_2[5, 5] = H_2nd_nn_2[2, 2]
        H_2nd_nn_2[5, 6] = H_2nd_nn_2[2, 3]
        H_2nd_nn_2[5, 7] = H_2nd_nn_2[2, 4]
        
        # Row 6
        H_2nd_nn_2[6, 1] = -H_2nd_nn_2[1, 6]
        H_2nd_nn_2[6, 5] = H_2nd_nn_2[3, 2]
        H_2nd_nn_2[6, 6] = H_2nd_nn_2[3, 3]
        H_2nd_nn_2[6, 7] = H_2nd_nn_2[3, 4]
        
        # Row 7
        H_2nd_nn_2[7, 1] = -H_2nd_nn_2[1, 7]
        H_2nd_nn_2[7, 5] = H_2nd_nn_2[4, 2]
        H_2nd_nn_2[7, 6] = H_2nd_nn_2[4, 3]
        H_2nd_nn_2[7, 7] = H_2nd_nn_2[4, 4]
        
        return H_2nd_nn_2


    def get_H_2nd_nn_3_Bi_Sb(self, k):
        # Initiate zero matrix #
        H_2nd_nn_3 = np.zeros((self.n_orbitals, self.n_orbitals), dtype=complex)
        
        ### Fill matrix with valuself.Es ###
        
        # Row 0
        H_2nd_nn_3[0, 0] = self.get_g13_3(k) * self.V2_ss_sigma_Bi_Sb
        H_2nd_nn_3[0, 3] = self.get_g15_3(k) * self.V2_sp_sigma_Bi_Sb
        H_2nd_nn_3[0, 4] = self.get_g16_3(k) * self.V2_sp_sigma_Bi_Sb
        
        # Row 1
        H_2nd_nn_3[1, 1] = self.get_g13_3(k) * self.V2_ss_sigma_Bi_Sb
        H_2nd_nn_3[1, 6] = self.get_g15_3(k) * self.V2_sp_sigma_Bi_Sb
        H_2nd_nn_3[1, 7] = self.get_g16_3(k) * self.V2_sp_sigma_Bi_Sb
        
        # Row 2
        H_2nd_nn_3[2, 0] = -H_2nd_nn_3[0, 2]
        H_2nd_nn_3[2, 2] = self.get_g18_3(k) * self.V2_pp_pi_Bi_Sb
        
        # Row 3
        H_2nd_nn_3[3, 0] = -H_2nd_nn_3[0, 3]
        H_2nd_nn_3[3, 2] = H_2nd_nn_3[2, 3]
        H_2nd_nn_3[3, 3] = self.get_g20_3(k) * self.V2_pp_sigma_Bi_Sb + self.get_g21_3(k) * self.V2_pp_pi_Bi_Sb
        H_2nd_nn_3[3, 4] = self.get_g24_3(k) * (self.V2_pp_sigma_Bi_Sb - self.V2_pp_pi_Bi_Sb)
        
        # Row 4
        H_2nd_nn_3[4, 0] = -H_2nd_nn_3[0, 4]
        H_2nd_nn_3[4, 2] = H_2nd_nn_3[2, 4]
        H_2nd_nn_3[4, 3] = H_2nd_nn_3[3, 4]
        H_2nd_nn_3[4, 4] = self.get_g22_3(k) * self.V2_pp_sigma_Bi_Sb + self.get_g23_3(k) * self.V2_pp_pi_Bi_Sb
        
        # Row 5
        H_2nd_nn_3[5, 1] = -H_2nd_nn_3[1, 5]
        H_2nd_nn_3[5, 5] = H_2nd_nn_3[2, 2]
        H_2nd_nn_3[5, 6] = H_2nd_nn_3[2, 3]
        H_2nd_nn_3[5, 7] = H_2nd_nn_3[2, 4]
        
        # Row 6
        H_2nd_nn_3[6, 1] = -H_2nd_nn_3[1, 6]
        H_2nd_nn_3[6, 5] = H_2nd_nn_3[3, 2]
        H_2nd_nn_3[6, 6] = H_2nd_nn_3[3, 3]
        H_2nd_nn_3[6, 7] = H_2nd_nn_3[3, 4]
        
        # Row 7
        H_2nd_nn_3[7, 1] = -H_2nd_nn_3[1, 7]
        H_2nd_nn_3[7, 5] = H_2nd_nn_3[4, 2]
        H_2nd_nn_3[7, 6] = H_2nd_nn_3[4, 3]
        H_2nd_nn_3[7, 7] = H_2nd_nn_3[4, 4]
        
        return H_2nd_nn_3


    def get_H_3rd_nn_1_Bi_Sb(self, k):
        # Initiate zero matrix #
        H_3rd_nn_1 = np.zeros((self.n_orbitals, self.n_orbitals), dtype=complex)
        
        ### Fill upper triangular part of matrix with valuself.Es ###
        
        # Row 0
        H_3rd_nn_1[0, 0] = self.get_g26_1(k) * self.V3_ss_sigma_Bi_Sb
        H_3rd_nn_1[0, 2] = self.get_g27_1(k) * self.V3_sp_sigma_Bi_Sb
        H_3rd_nn_1[0, 3] = self.get_g28_1(k) * self.V3_sp_sigma_Bi_Sb
        
        # Row 1
        H_3rd_nn_1[1, 1] = self.get_g26_1(k) * self.V3_ss_sigma_Bi_Sb
        H_3rd_nn_1[1, 5] = self.get_g27_1(k) * self.V3_sp_sigma_Bi_Sb
        H_3rd_nn_1[1, 6] = self.get_g28_1(k) * self.V3_sp_sigma_Bi_Sb
        
        # Row 2
        H_3rd_nn_1[2, 2] = self.get_g29_1(k) * self.V3_pp_sigma_Bi_Sb + self.get_g30_1(k) * self.V3_pp_pi_Bi_Sb
        H_3rd_nn_1[2, 3] = self.get_g31_1(k) * (self.V3_pp_sigma_Bi_Sb - self.V3_pp_pi_Bi_Sb)
        
        # Row 3
        H_3rd_nn_1[3, 3] = self.get_g30_1(k) * self.V3_pp_sigma_Bi_Sb + self.get_g29_1(k) * self.V3_pp_pi_Bi_Sb
        
        # Row 4
        H_3rd_nn_1[4, 4] = self.get_g26_1(k) * self.V3_pp_pi_Bi_Sb
        
        # Row 5
        H_3rd_nn_1[5, 5] = self.get_g29_1(k) * self.V3_pp_sigma_Bi_Sb + self.get_g30_1(k) * self.V3_pp_pi_Bi_Sb
        H_3rd_nn_1[5, 6] = self.get_g31_1(k) * (self.V3_pp_sigma_Bi_Sb - self.V3_pp_pi_Bi_Sb)
        
        # Row 6
        H_3rd_nn_1[6, 6] = self.get_g30_1(k) * self.V3_pp_sigma_Bi_Sb + self.get_g29_1(k) * self.V3_pp_pi_Bi_Sb
        
        # Row 7
        H_3rd_nn_1[7, 7] = self.get_g26_1(k) * self.V3_pp_pi_Bi_Sb
        
        # Add the hermitian conjugate of the upper triangular to get hermitian matrix #
        H_3rd_nn_1 += np.transpose(np.triu(H_3rd_nn_1, k=1))
        
        return H_3rd_nn_1


    def get_H_3rd_nn_2_Bi_Sb(self, k):
        # Initiate zero matrix #
        H_3rd_nn_2 = np.zeros((self.n_orbitals, self.n_orbitals), dtype=complex)
        
        ### Fill upper triangular part of matrix with valuself.Es ###
        
        # Row 0
        H_3rd_nn_2[0, 0] = self.get_g26_2(k) * self.V3_ss_sigma_Bi_Sb
        H_3rd_nn_2[0, 2] = self.get_g27_2(k) * self.V3_sp_sigma_Bi_Sb
        H_3rd_nn_2[0, 3] = self.get_g28_2(k) * self.V3_sp_sigma_Bi_Sb
        
        # Row 1
        H_3rd_nn_2[1, 1] = self.get_g26_2(k) * self.V3_ss_sigma_Bi_Sb
        H_3rd_nn_2[1, 5] = self.get_g27_2(k) * self.V3_sp_sigma_Bi_Sb
        H_3rd_nn_2[1, 6] = self.get_g28_2(k) * self.V3_sp_sigma_Bi_Sb
        
        # Row 2
        H_3rd_nn_2[2, 2] = self.get_g29_2(k) * self.V3_pp_sigma_Bi_Sb + self.get_g30_2(k) * self.V3_pp_pi_Bi_Sb
        H_3rd_nn_2[2, 3] = self.get_g31_2(k) * (self.V3_pp_sigma_Bi_Sb - self.V3_pp_pi_Bi_Sb)
        
        # Row 3
        H_3rd_nn_2[3, 3] = self.get_g30_2(k) * self.V3_pp_sigma_Bi_Sb + self.get_g29_2(k) * self.V3_pp_pi_Bi_Sb
        
        # Row 4
        H_3rd_nn_2[4, 4] = self.get_g26_2(k) * self.V3_pp_pi_Bi_Sb
        
        # Row 5
        H_3rd_nn_2[5, 5] = self.get_g29_2(k) * self.V3_pp_sigma_Bi_Sb + self.get_g30_2(k) * self.V3_pp_pi_Bi_Sb
        H_3rd_nn_2[5, 6] = self.get_g31_2(k) * (self.V3_pp_sigma_Bi_Sb - self.V3_pp_pi_Bi_Sb)
        
        # Row 6
        H_3rd_nn_2[6, 6] = self.get_g30_2(k) * self.V3_pp_sigma_Bi_Sb + self.get_g29_2(k) * self.V3_pp_pi_Bi_Sb
        
        # Row 7
        H_3rd_nn_2[7, 7] = self.get_g26_2(k) * self.V3_pp_pi_Bi_Sb
        
        # Add the hermitian conjugate of the upper triangular to get hermition matrix #
        H_3rd_nn_2 += np.transpose(np.triu(H_3rd_nn_2, k=1))
        
        return H_3rd_nn_2


    def get_H_3rd_nn_3_Bi_Sb(self, k):
        # Initiate zero matrix #
        H_3rd_nn_3 = np.zeros((self.n_orbitals, self.n_orbitals), dtype=complex)
        
        ### Fill upper triangular part of matrix with valuself.Es ###
        
        # Row 0
        H_3rd_nn_3[0, 0] = self.get_g26_3(k) * self.V3_ss_sigma_Bi_Sb
        H_3rd_nn_3[0, 2] = self.get_g27_3(k) * self.V3_sp_sigma_Bi_Sb
        
        # Row 1
        H_3rd_nn_3[1, 1] = self.get_g26_3(k) * self.V3_ss_sigma_Bi_Sb
        H_3rd_nn_3[1, 5] = self.get_g27_3(k) * self.V3_sp_sigma_Bi_Sb
        
        # Row 2
        H_3rd_nn_3[2, 2] = self.get_g29_3(k) * self.V3_pp_sigma_Bi_Sb
        
        # Row 3
        H_3rd_nn_3[3, 3] = self.get_g29_3(k) * self.V3_pp_pi_Bi_Sb
        
        # Row 4
        H_3rd_nn_3[4, 4] = self.get_g26_3(k) * self.V3_pp_pi_Bi_Sb
        
        # Row 5
        H_3rd_nn_3[5, 5] = self.get_g29_3(k) * self.V3_pp_sigma_Bi_Sb
        
        # Row 6
        H_3rd_nn_3[6, 6] = self.get_g29_3(k) * self.V3_pp_pi_Bi_Sb
        
        # Row 7
        H_3rd_nn_3[7, 7] = self.get_g26_3(k) * self.V3_pp_pi_Bi_Sb
        
        # Add the hermitian conjugate of the upper triangular to get hermition matrix #
        H_3rd_nn_3 += np.transpose(np.triu(H_3rd_nn_3, k=1))
        
        return H_3rd_nn_3


    def get_H_all_nn_Bi_Sb(self):
        return [
            [self.get_H_1st_nn_1_Bi_Sb, self.get_H_1st_nn_2_Bi_Sb, self.get_H_1st_nn_3_Bi_Sb],
            [self.get_H_2nd_nn_1_Bi_Sb, self.get_H_2nd_nn_2_Bi_Sb, self.get_H_2nd_nn_3_Bi_Sb],
            [self.get_H_3rd_nn_1_Bi_Sb, self.get_H_3rd_nn_2_Bi_Sb, self.get_H_3rd_nn_3_Bi_Sb]
            ]


class ModelSlab(ModelCrystal):

    def __init__(self, crystal_c_v1, crystal_c_v2, n_layers, n_extend_x, n_extend_y, n_atoms_removed, atom_types,
                 slab_type, prefix, kwargs=None):
        """
        Defines a Bi, Sb or BiSb (surface_normal e.g. 112 or 111)-slab that is produced by repeating the Liu-Allen geometry (lattice_const_a/c) of 4 atoms n_layers times
        in the z-direction and n_extend_x/y in the x/y-directions. n_layers should be an even number because of how get_edge_participation is
        implemented. n_atoms_removed is for removing spcific atoms from the generated crystal to generate alternative surfaces (truncated bulk
        if n_atoms_removed=0, reconstruction I (II) with 1 (2) atoms removed). atom_types is for defining the crystal geomtry (either Bi or Sb
        from Liu-Allen) and for choosing the working folder for storing/reading files. Supported values of slab_type is currently {truncated_bulk,
        truncated_bulk_BiSb, crystal17 (aka lines with stochiometry Bi13Sb3), reconstruction_typeI, reconstruction_typeII}. kwargs is for defining
        the BiSb hopping params from those of Bi and Sb (factor_1, scale_Bi_Sb = kwargs). After initializing the model, one typically runs
        define_atom_types to generate a list with atom numbers and atom types, it is stored in the atom_types folder. Then one can run
        define_hopping_matrix to generate a list containing all atom numbers, all their neighbouring atom numbers, two numbers that descibe 1st, 2nd
        or 3rd neighbour hopping plus the hopping vector (a1,a2 or a3), and the atom types (Bi or Sb). The hopping matrix is stored as a text file
        in the hopping_matrix_folder. Then one can run calculate_bandstructure along a high symmetry path or calculate_constant_energy_surface to
        solve the hopping matrices for certain values of k, see details in the functions. Lastly one can plot the band structure using plot_band_structure
        or plot_constant_energy_surface, choosing parameters for plotting such as spin exp values or edge participation.
        """

        #############################################################################
        ### Define the crystal lattice parameters and the hopping matrix elements ###
        #############################################################################

        # Class definition parameters
        super().__init__(atom_types, prefix)
        self.n_layers = n_layers
        self.n_extend_x = n_extend_x
        self.n_extend_y = n_extend_y
        self.n_atoms_removed = n_atoms_removed
        self.slab_type = slab_type

        # Total number of atoms in the unit cell (! generalize this to other than 4 atoms !)
        self.n_atoms = 4 * self.n_layers * self.n_extend_x * self.n_extend_y - self.n_atoms_removed
        
        # Number of localized atomic orbitals per atom (including spin)
        self.n_bnd = self.n_orbitals * self.n_atoms

        # Slab lattice constant
        self.slab_lattice_const_c = 10 * (self.n_layers + 1)

        # Slab lattice vectors
        self.slab_a1 = np.dot(crystal_c_v1, self.bulk_lattice_vectors) * self.n_extend_x
        self.slab_a2 = np.dot(crystal_c_v2, self.bulk_lattice_vectors) * self.n_extend_y
        self.slab_a3 = np.cross(self.slab_a2, self.slab_a1)
        self.slab_a3 /= np.linalg.norm(self.slab_a3)
        self.slab_a3 *= self.slab_lattice_const_c

        # Slab reciprocal lattice vectors
        V = np.dot(self.slab_a1, np.cross(self.slab_a2, self.slab_a3))
        self.slab_b1 = (2 * np.pi / V) * np.cross(self.slab_a2, self.slab_a3)
        self.slab_b2 = (2 * np.pi / V) * np.cross(self.slab_a3, self.slab_a1)
        self.slab_b3 = (2 * np.pi / V) * np.cross(self.slab_a1, self.slab_a2)

    def get_hamiltonian(self, k, file_name_hopping_matrix):
        ### Define all the individual hopping matrices "H_n_n" of the upper triangular part of H ###

        # Get single-site hopping matrices
        H_all_nn_Bi = self.get_H_all_nn_Bi()
        H_all_nn_Sb = self.get_H_all_nn_Sb()
        H_all_nn_Bi_Sb = self.get_H_all_nn_Bi_Sb()

        # Initiate zero matrix
        H = np.zeros((self.n_bnd, self.n_bnd), dtype=complex)

        # Read hopping matrix from file
        hopping_matrix = open(self.parent_folder + file_name_hopping_matrix, 'r')

        # Loop through hopping terms in the matrix
        for hopping_element_str in hopping_matrix:

            # Skip empty lines
            if hopping_element_str == '\n':
                continue

                # Read line in hopping matrix, extract atom numbers and types
            hopping_element = np.array(hopping_element_str[:-7].split(' '), dtype=int)
            atom1 = hopping_element[0] - 1
            atom2 = hopping_element[1] - 1
            atom_type1 = hopping_element_str[-6:-4]
            atom_type2 = hopping_element_str[-3:-1]

            # Check if hopping is Bi-Bi, Sb-Sb, or Bi-Sb
            if atom_type1 == "Bi" and atom_type2 == "Bi":
                if hopping_element[2] == 0:
                    # Onsite hopping
                    element = self.get_H_onsite_Bi()
                else:
                    # Sign determines direction of hopping, then choose the correct single-site hopping term (e.g. 1st-NN-2)
                    element_sign = np.sign(hopping_element[3])
                    if element_sign == 1:
                        element = H_all_nn_Bi[hopping_element[2] - 1][abs(hopping_element[3]) - 1](k)
                    else:
                        element = np.transpose(
                            np.conjugate(H_all_nn_Bi[hopping_element[2] - 1][abs(hopping_element[3]) - 1](k)))

            elif atom_type1 == "Sb" and atom_type2 == "Sb":
                if hopping_element[2] == 0:
                    # Onsite hopping
                    element = self.get_H_onsite_Sb()
                else:
                    # Sign determines direction of hopping, then choose the correct single-site hopping term (e.g. 1st-NN-2)
                    element_sign = np.sign(hopping_element[3])
                    if element_sign == 1:
                        element = H_all_nn_Sb[hopping_element[2] - 1][abs(hopping_element[3]) - 1](k)
                    else:
                        element = np.transpose(
                            np.conjugate(H_all_nn_Sb[hopping_element[2] - 1][abs(hopping_element[3]) - 1](k)))


            elif atom_type1 == "Bi" and atom_type2 == "Sb" or atom_type1 == "Sb" and atom_type2 == "Bi":
                if hopping_element[2] == 0:
                    # Onsite hopping Bi-Sb not possible
                    raise Exception("Onsite hopping between Bi and Sb")
                else:
                    # Sign determines direction of hopping, then choose the correct single-site hopping term (e.g. 1st-NN-2)
                    element_sign = np.sign(hopping_element[3])
                    if element_sign == 1:
                        element = H_all_nn_Bi_Sb[hopping_element[2] - 1][abs(hopping_element[3]) - 1](k)
                    else:
                        element = np.transpose(
                            np.conjugate(H_all_nn_Bi_Sb[hopping_element[2] - 1][abs(hopping_element[3]) - 1](k)))

            else:
                raise Exception("Atom type not Bi or Sb")

            # Add the hopping term for this element in the hopping matrix loop
            H[atom1 * self.n_orbitals:(atom1 + 1) * self.n_orbitals,
            atom2 * self.n_orbitals:(atom2 + 1) * self.n_orbitals] += element

        # Close text file
        hopping_matrix.close()

        # Throw Exception if H is not Hermitian
        if not np.isclose(np.transpose(np.conjugate(H)), H).all():
            # print(np.where(np.not_equal(np.isclose(np.transpose(np.conjugate(H)), H), True)))
            raise Exception("H is not hermitian")

        return H

    def get_edge_participation(self, eigvec):
        # Returns a vector containing the edge participation of each eigenfunction at a specific k-point

        # Convert amplitudes to probabilities
        probabilities = np.square(np.abs(eigvec))

        # Make exp decay vector
        # Requires an even number of self.n_layers
        exp_decay_from_surface = np.concatenate((np.linspace(1, self.n_layers // 2, self.n_layers // 2),
                                                 np.linspace(self.n_layers // 2, 1, self.n_layers // 2)))
        exp_decay_from_surface = np.repeat(exp_decay_from_surface, 4)

        # Repeat the exp decay pattern self.n_extend_x * self.n_extend_y times
        exp_decay_from_surface = np.tile(exp_decay_from_surface, self.n_extend_x * self.n_extend_y)

        # If there are atoms removed (not truncated bulk) then delete these atoms from the exp decay pattern
        if self.slab_type == "reconstruction_typeI":
            if self.n_extend_y == 1:
                self.remove_atoms_numbers = np.array([3 + 4 * (self.n_layers - 1), 4]) - 1
            elif self.n_extend_y == 2:
                self.remove_atoms_numbers = np.concatenate((np.array([1, 2, 3, 3 + 4 * self.n_layers]) + 4 * (
                            self.n_layers - 1), np.array([4, 4 + 4 * self.n_layers, 2, 1]))) - 1
            elif self.n_extend_y == 3:
                self.remove_atoms_numbers_prev = np.concatenate((np.array([1, 2, 3, 3 + 4 * self.n_layers]) + 4 * (
                            self.n_layers - 1), np.array([4, 4 + 4 * self.n_layers, 2, 1]))) - 1
                self.remove_atoms_numbers = np.concatenate((self.remove_atoms_numbers_prev,
                                                            self.remove_atoms_numbers_prev + 4 * self.n_layers,
                                                            np.array([3 + 4 * self.n_layers,
                                                                      4 + 4 * self.n_layers + 4 * (self.n_layers - 1),
                                                                      3 + 4 * (self.n_layers - 1) * 2,
                                                                      4 + 4 * (self.n_layers + 1)]) - 1))
            elif self.n_extend_y == 4:
                self.remove_atoms_numbers_prev = np.concatenate((np.array([1, 2, 3, 3 + 4 * self.n_layers]) + 4 * (
                            self.n_layers - 1), np.array([4, 4 + 4 * self.n_layers, 2, 1]))) - 1
                self.remove_atoms_numbers_prev = np.concatenate((self.remove_atoms_numbers_prev,
                                                                 self.remove_atoms_numbers_prev + 4 * self.n_layers,
                                                                 np.array([3 + 4 * self.n_layers,
                                                                           4 + 4 * self.n_layers + 4 * (
                                                                                       self.n_layers - 1),
                                                                           3 + 4 * (self.n_layers - 1) * 2,
                                                                           4 + 4 * (self.n_layers + 1)]) - 1))
                self.remove_atoms_numbers = np.concatenate((self.remove_atoms_numbers_prev,
                                                            self.remove_atoms_numbers_prev + 4 * self.n_layers,
                                                            np.array([1 + 4 * (self.n_layers + 1),
                                                                      2 + 4 * (self.n_layers + 1),
                                                                      1 + 2 * 4 * (self.n_layers - 1),
                                                                      2 + 2 * 4 * (self.n_layers - 1)]) - 1))

            exp_decay_from_surface = np.delete(exp_decay_from_surface, np.unique(self.remove_atoms_numbers))

        if self.slab_type == "reconstruction_typeII":
            if self.n_extend_y == 2:
                self.remove_atoms_numbers = np.concatenate(
                    (np.array([1, 3]) + 4 * (self.n_layers - 1), np.array([4, 2 + 4 * self.n_layers]))) - 1
            elif self.n_extend_y == 3:
                self.remove_atoms_numbers_prev = np.concatenate(
                    (np.array([1, 3]) + 4 * (self.n_layers - 1), np.array([2, 4 + 4 * self.n_layers]))) - 1
                self.remove_atoms_numbers = np.concatenate((self.remove_atoms_numbers_prev,
                                                            self.remove_atoms_numbers_prev + 4 * self.n_layers,
                                                            np.array([2 + 4 * (self.n_layers - 1),
                                                                      4 + 4 * (2 * self.n_layers - 1),
                                                                      3 + 4 * self.n_layers,
                                                                      1 + 4 * self.n_layers]) - 1))
            elif self.n_extend_y == 4:
                self.remove_atoms_numbers_prev = np.concatenate(
                    (np.array([1, 3]) + 4 * (self.n_layers - 1), np.array([2, 4 + 4 * self.n_layers]))) - 1
                self.remove_atoms_numbers_prev = np.concatenate((self.remove_atoms_numbers_prev,
                                                                 self.remove_atoms_numbers_prev + 4 * self.n_layers,
                                                                 np.array([2 + 4 * (self.n_layers - 1),
                                                                           4 + 4 * (2 * self.n_layers - 1),
                                                                           3 + 4 * self.n_layers,
                                                                           1 + 4 * self.n_layers]) - 1))
                self.remove_atoms_numbers = np.concatenate((self.remove_atoms_numbers_prev,
                                                            self.remove_atoms_numbers_prev + 4 * self.n_layers,
                                                            np.array([1 + 4 * (2 * self.n_layers - 2),
                                                                      3 + 4 * (2 * self.n_layers - 2),
                                                                      2 + 4 * (self.n_layers + 1),
                                                                      4 + 4 * (2 * self.n_layers + 1)]) - 1))

            exp_decay_from_surface = np.delete(exp_decay_from_surface, np.unique(self.remove_atoms_numbers))

        # Repeat for the number of orbitals
        exp_decay_from_surface = np.repeat(exp_decay_from_surface, self.n_orbitals)

        # Make it decay exponentially
        exp_decay_from_surface = np.exp(1 - exp_decay_from_surface)

        # Make exp decay matrix
        shape = exp_decay_from_surface.shape
        exp_decay_from_surface = np.reshape(np.repeat(exp_decay_from_surface, probabilities.shape[1]),
                                            (*shape, probabilities.shape[1]))

        # Find edge state participation by multiplying prob by exp decay from surface
        edge_participation = probabilities * exp_decay_from_surface

        return np.sum(edge_participation, axis=0)

    def get_edge_participation_one_sided(self, eigvec, select_n_layers):
        # Returns a vector containing the edge participation at ONE slab surface of each eigenfunction at a specific k-point
        # (!) NOT IMPLEMENTED for reconstruction I or II (ie any atoms removed)
        # This function does not use exponential decay like the two-sided version, it just returns the sum of the pobability
        # in the first select_n_layers, which also seems to work fine to isolate the SS on one side

        # Convert amplitudes to probabilities
        probabilities = np.square(np.abs(eigvec))
        probabilities = probabilities.reshape((self.n_bnd, self.n_bnd))

        edge_participation = []
        # Sum the probabilities in the first select_n_layers
        for i in range(self.n_bnd):

            # Sum of probabilities if n_extend_x=1 and n_extend_y=1
            sum_probabilities = np.sum(probabilities[:4 * select_n_layers * self.n_orbitals, i])

            # Add the probabilities of the extensions if n_extend_x>1 and/or n_extend_y>1
            for j in range(1, self.n_extend_x * self.n_extend_y):
                sum_probabilities += np.sum(probabilities[
                                            j * 4 * self.n_layers * self.n_orbitals:j * 4 * self.n_layers * self.n_orbitals + 4 * select_n_layers * self.n_orbitals,
                                            i])

            # Append to list
            edge_participation.append(sum_probabilities)

        return np.array(edge_participation)

    def get_spin_expectation_values(self, current_eigvec):
        # Return a matrix containing the spin expectation values (Sx,Sy,Sz) of all bands for a given eigenvector (ie given k-point)
        eigvec = current_eigvec.reshape((self.n_bnd, self.n_bnd))

        # Define Pauli matrices
        spin_x = np.array([[0, 1], [1, 0]])
        spin_y = np.array([[0, -1j], [1j, 0]])
        spin_z = np.array([[1, 0], [0, -1]])
        pauli_vector = np.array([spin_x, spin_y, spin_z])

        # Rotate coordinate system to lie along lattice vectors (a1,a2,a3)
        x_dir = self.slab_a1 / np.linalg.norm(self.slab_a1)
        y_dir = self.slab_a2 / np.linalg.norm(self.slab_a2)
        z_dir = self.slab_a3 / np.linalg.norm(self.slab_a3)

        # Rotate Pauli vector to lie along lattice vectors (a1,a2,a3)
        spin_x = np.tensordot(x_dir, pauli_vector, axes=([0], [0]))
        spin_y = np.tensordot(y_dir, pauli_vector, axes=([0], [0]))
        spin_z = np.tensordot(z_dir, pauli_vector, axes=([0], [0]))

        # Tensor product (Kronecker) to get Pauli matrix for each orbital (4 x 2spin)
        spin_x = np.kron(np.eye(4), spin_x)
        spin_y = np.kron(np.eye(4), spin_y)
        spin_z = np.kron(np.eye(4), spin_z)

        # Permute ordering (s and p orbitals)
        permutation = [0, 1, 2, 5, 3, 6, 4, 7]
        idx = np.empty_like(permutation)
        idx[permutation] = np.arange(len(permutation))

        spin_x = spin_x[:, idx]
        spin_y = spin_y[:, idx]
        spin_z = spin_z[:, idx]

        spin_x = spin_x[idx, :]
        spin_y = spin_y[idx, :]
        spin_z = spin_z[idx, :]

        # Tensor product (Kronecker) to get Pauli matrix for each atom
        spin_x = np.kron(np.eye(self.n_bnd // 8), spin_x)
        spin_y = np.kron(np.eye(self.n_bnd // 8), spin_y)
        spin_z = np.kron(np.eye(self.n_bnd // 8), spin_z)

        # Calc exp value of each spin component <Psi|S_i|Psi>
        expectation_values = []
        for i in range(self.n_bnd):
            exp_x = np.real(np.dot(np.conj(eigvec[:, i]), np.dot(spin_x, eigvec[:, i])))
            exp_y = np.real(np.dot(np.conj(eigvec[:, i]), np.dot(spin_y, eigvec[:, i])))
            exp_z = np.real(np.dot(np.conj(eigvec[:, i]), np.dot(spin_z, eigvec[:, i])))
            expectation_values.append([exp_x, exp_y, exp_z])

        return np.array(expectation_values)

    def define_atom_types(self):
        # Available slab types are truncated_bulk, truncated_bulk_BiSb, crystal17,
        # reconstruction_typeI, reconstruction_typeII

        match self.slab_type:
            case "truncated_bulk":
                # Write pure truncated Bi or Sb bulk atom types to file
                file = open(
                    self.parent_folder + "atom_types/atom_types_" + self.prefix, "w")
                file.write("ID Atom_type\n")
                for atom_nr in range(self.n_atoms):
                    file.write(str(atom_nr + 1) + " " + self.atom_types + "\n")
                file.close()

            case "truncated_bulk_BiSb":
                # Write BiSb alloy truncated bulk atom types to file
                file = open(
                    self.parent_folder + "atom_types/atom_types_" + self.prefix, "w")
                file.write("ID Atom_type\n")
                for atom_nr in range(self.n_atoms):
                    if atom_nr % 4 == 0:
                        file.write(str(atom_nr + 1) + " " + "Sb" + "\n")
                    else:
                        file.write(str(atom_nr + 1) + " " + "Bi" + "\n")
                file.close()

            case "crystal17":
                # Write crystal number 17 (aka lines) to file
                # Is defined for n_x = 2 and n_y = 2 (for n_x, n_y = 1 you get Bi3Sb1)
                file = open(
                    self.parent_folder + "atom_types/atom_types_" + self.prefix, "w")
                file.write("ID Atom_type\n")
                for atom_nr in range(self.n_atoms):

                    # First unit cell
                    if (atom_nr + 1) // (4 * self.n_layers) == 0:
                        if (atom_nr + 1) % 8 == 3 or (atom_nr + 1) % 8 == 5:
                            file.write(str(atom_nr + 1) + " " + "Sb" + "\n")
                            continue

                    # Second unit cell
                    if (atom_nr + 1) // (4 * self.n_layers) == 1:
                        if ((atom_nr + 1) - 4 * self.n_layers) % 8 == 3:
                            file.write(str(atom_nr + 1) + " " + "Sb" + "\n")
                            continue

                    # Third unit cell
                    if (atom_nr + 1) // (4 * self.n_layers) == 2:
                        if ((atom_nr + 1) - 2 * 4 * self.n_layers) % 8 == 7:
                            file.write(str(atom_nr + 1) + " " + "Sb" + "\n")
                            continue

                    # Fourth unit cell
                    if (atom_nr + 1) // (4 * self.n_layers) == 3:
                        if ((atom_nr + 1) - 3 * 4 * self.n_layers) % 8 == 7 or (
                                (atom_nr + 1) - 3 * 4 * self.n_layers) % 8 == 1:
                            file.write(str(atom_nr + 1) + " " + "Sb" + "\n")
                            continue

                    file.write(str(atom_nr + 1) + " " + "Bi" + "\n")
                file.close()

            case "reconstruction_typeI":
                # Write pure reconstructed (atoms removed) atom types to file -- (RECONSTRUCTION TYPE I)

                if self.n_extend_y == 1:
                    remove_atoms_numbers = np.array([3 + 4 * (self.n_layers - 1), 4]) - 1
                elif self.n_extend_y == 2:
                    remove_atoms_numbers = np.concatenate(
                        (np.array([1, 2, 3, 3 + 4 * self.n_layers]) + 4 * (self.n_layers - 1),
                         np.array([4, 4 + 4 * self.n_layers, 2, 1]))) - 1
                elif self.n_extend_y == 3:
                    remove_atoms_numbers_prev = np.concatenate((np.array([1, 2, 3, 3 + 4 * self.n_layers]) + 4 * (
                            self.n_layers - 1), np.array([4, 4 + 4 * self.n_layers, 2, 1]))) - 1
                    remove_atoms_numbers = np.concatenate((remove_atoms_numbers_prev,
                                                           remove_atoms_numbers_prev + 4 * self.n_layers, np.array(
                        [3 + 4 * self.n_layers, 4 + 4 * self.n_layers + 4 * (self.n_layers - 1),
                         3 + 4 * (self.n_layers - 1) * 2,
                         4 + 4 * (self.n_layers + 1)]) - 1))
                elif self.n_extend_y == 4:
                    remove_atoms_numbers_prev = np.concatenate((np.array([1, 2, 3, 3 + 4 * self.n_layers]) + 4 * (
                            self.n_layers - 1), np.array([4, 4 + 4 * self.n_layers, 2, 1]))) - 1
                    remove_atoms_numbers_prev = np.concatenate((remove_atoms_numbers_prev,
                                                                remove_atoms_numbers_prev + 4 * self.n_layers, np.array(
                        [3 + 4 * self.n_layers, 4 + 4 * self.n_layers + 4 * (self.n_layers - 1),
                         3 + 4 * (self.n_layers - 1) * 2,
                         4 + 4 * (self.n_layers + 1)]) - 1))
                    remove_atoms_numbers = np.concatenate((remove_atoms_numbers_prev,
                                                           remove_atoms_numbers_prev + 4 * self.n_layers, np.array(
                        [1 + 4 * (self.n_layers + 1), 2 + 4 * (self.n_layers + 1), 1 + 2 * 4 * (self.n_layers - 1),
                         2 + 2 * 4 * (self.n_layers - 1)]) - 1))
                else:
                    print("No added periodicity within range (n_extend_y = 1-4)")
                    return None

                remove_atoms_numbers = np.unique(remove_atoms_numbers)

                file = open(
                    self.parent_folder + "atom_types/atom_types_" + self.prefix, "w")

                file.write("ID Atom_type\n")
                for atom_nr in range(self.n_atoms + self.n_atoms_removed):
                    if atom_nr in remove_atoms_numbers:
                        file.write(str(atom_nr + 1) + " " + "removed" + "\n")
                    else:
                        file.write(str(atom_nr + 1) + " " + self.atom_types + "\n")
                file.close()

            case "reconstruction_typeII":
                # Write pure reconstructed (atoms removed) atom types to file -- (RECONSTRUCTION TYPE II)

                if self.n_extend_y == 2:
                    remove_atoms_numbers = np.concatenate(
                        (np.array([1, 3]) + 4 * (self.n_layers - 1), np.array([4, 2 + 4 * self.n_layers]))) - 1
                elif self.n_extend_y == 3:
                    remove_atoms_numbers_prev = np.concatenate(
                        (np.array([1, 3]) + 4 * (self.n_layers - 1), np.array([2, 4 + 4 * self.n_layers]))) - 1
                    remove_atoms_numbers = np.concatenate((remove_atoms_numbers_prev,
                                                           remove_atoms_numbers_prev + 4 * self.n_layers, np.array(
                        [2 + 4 * (self.n_layers - 1), 4 + 4 * (2 * self.n_layers - 1), 3 + 4 * self.n_layers,
                         1 + 4 * self.n_layers]) - 1))
                elif self.n_extend_y == 4:
                    remove_atoms_numbers_prev = np.concatenate(
                        (np.array([1, 3]) + 4 * (self.n_layers - 1), np.array([2, 4 + 4 * self.n_layers]))) - 1
                    remove_atoms_numbers_prev = np.concatenate((remove_atoms_numbers_prev,
                                                                remove_atoms_numbers_prev + 4 * self.n_layers, np.array(
                        [2 + 4 * (self.n_layers - 1), 4 + 4 * (2 * self.n_layers - 1), 3 + 4 * self.n_layers,
                         1 + 4 * self.n_layers]) - 1))
                    remove_atoms_numbers = np.concatenate((remove_atoms_numbers_prev,
                                                           remove_atoms_numbers_prev + 4 * self.n_layers, np.array(
                        [1 + 4 * (2 * self.n_layers - 2), 3 + 4 * (2 * self.n_layers - 2), 2 + 4 * (self.n_layers + 1),
                         4 + 4 * (2 * self.n_layers + 1)]) - 1))
                else:
                    print("No added periodicity within range (n_extend_y = 2-4)")
                    return None

                remove_atoms_numbers = np.unique(remove_atoms_numbers)

                file = open(
                    self.parent_folder + "atom_types/atom_types_" + self.prefix, "w")

                file.write("ID Atom_type\n")
                for atom_nr in range(self.n_atoms + self.n_atoms_removed):
                    if atom_nr in remove_atoms_numbers:
                        file.write(str(atom_nr + 1) + " " + "removed" + "\n")
                    else:
                        file.write(str(atom_nr + 1) + " " + self.atom_types + "\n")
                file.close()

            case other:
                print("Error: slab type not defined")
                return None

        print("Atoms types successfully written to file")
        return None

    def dist(self, i, j, unit_cell_atom_positions):
        v_i = unit_cell_atom_positions.iloc[i][["x", "y", "z"]]
        v_j = unit_cell_atom_positions.iloc[j][["x", "y", "z"]]

        min_dist = 1e6
        for x in [-1, 0, 1]:
            for y in [-1, 0, 1]:
                current_dist = np.linalg.norm(v_i - (v_j + self.slab_a1 * x + self.slab_a2 * y))
                if current_dist < min_dist:
                    min_dist = current_dist

        return min_dist

    def monte_carlo_distribution(self, m, atom_positions_file, crystal_c_v3, step_list, temperature_list):
        """
        Uses a Monte Carlo algorithm to find the subset of m points out of all lattice points
        that maximizes the average pairwise distance.

        Parameters:
        - m: int, number of points to select (ie Sb atoms)
        - dist: function, a function dist(i, j) that gives the distance between points i and j.

        Returns:
        - best_subset: list of int, indices of the points that maximize the average distance.
        - max_avg_min_distance: float, the maximum average distance achieved.
        """

        # Get the atom positions used for dist(i, j)
        unit_cell_atom_positions = self.define_hopping_matrix(atom_positions_file, crystal_c_v3,
                                                              return_unit_cell_atom_positions=True)

        def greedy_initialization():
            subset = [0]  # Start from the first point
            while len(subset) < m:
                # Choose the next point that maximizes the minimum distance to the points in `subset`
                best_next_point = None
                best_min_distance = 0
                for i in range(self.n_atoms):
                    if i in subset:
                        continue
                    # Calculate the minimum distance from point `i` to the current `subset`
                    closest_distance = min(self.dist(i, j, unit_cell_atom_positions) for j in subset if i != j)
                    if closest_distance > best_min_distance:
                        best_min_distance = closest_distance
                        best_next_point = i
                subset.append(best_next_point)
            print("d = " + str(average_min_distance(subset)))
            return subset

        # Function to calculate the average minimum distance in a given subset of points
        def average_min_distance(subset):
            min_distances = []
            for i in subset:
                # Find the closest distance to any other point in the subset
                closest_distance = min(self.dist(i, j, unit_cell_atom_positions) for j in subset if i != j)
                min_distances.append(closest_distance)
            # return min(min_distances)
            return sum(min_distances) / len(min_distances)

        current_subset = greedy_initialization()
        av_min_dist = average_min_distance(current_subset)
        for j in range(len(step_list)):
            count_list = np.array([0, 0])
            for i in range(step_list[j]):

                atom_to_move = random.randint(0, m - 1)
                new_loc = current_subset[0]
                while new_loc in current_subset:
                    new_loc = random.randint(0, self.n_atoms - 1)

                new_subset = current_subset.copy()
                new_subset[atom_to_move] = new_loc
                new_av_min_dist = average_min_distance(new_subset)

                if np.random.exponential() > (av_min_dist - new_av_min_dist) / temperature_list[j]:
                    current_subset = new_subset.copy()
                    av_min_dist = new_av_min_dist
                    count_list[0] += 1
                else:
                    count_list[1] += 1

            print(count_list)
            print(av_min_dist)
            print("  ")

        # Write BiSb alloy truncated bulk atom types to file
        file = open(
            self.parent_folder + "atom_types/atom_types_" + self.prefix, "w")
        file.write("ID Atom_type\n")
        for atom_nr in range(self.n_atoms):
            if atom_nr in current_subset:
                file.write(str(atom_nr + 1) + " " + "Sb" + "\n")
            else:
                file.write(str(atom_nr + 1) + " " + "Bi" + "\n")
        file.close()

        print(current_subset)
        print(average_min_distance(current_subset))

    def define_hopping_matrix(self, atom_positions_file, crystal_c_v3, return_unit_cell_atom_positions=False):
        ### Generate a text file containing all the atoms with types, and all neighbours (1st-3rd) ###

        # Read starting geometry (n atoms that will be repeated to create slab)
        unit_cell_atom_positions = pd.read_csv(
            self.parent_folder + "atom_positions/" + atom_positions_file, delim_whitespace=True)
        n_unit_cell_atoms = len(unit_cell_atom_positions)

        # This will be the vector containing all atoms (repeated in x-, y- and z-dir), now empty
        repeated_cells_atom_positions = pd.DataFrame(columns=unit_cell_atom_positions.columns)

        # Repeat in z-dir
        for i in range(self.n_layers):
            repeated_cells_atom_positions = pd.concat(
                [repeated_cells_atom_positions, unit_cell_atom_positions.iloc[0:n_unit_cell_atoms]], ignore_index=True)
            repeated_cells_atom_positions.iloc[-n_unit_cell_atoms:].loc[:, ["x", "y", "z"]] = \
            repeated_cells_atom_positions.iloc[-n_unit_cell_atoms:][["x", "y", "z"]] + np.dot(crystal_c_v3,
                                                                                              self.bulk_lattice_vectors) * i

        # Reset index
        unit_cell_atom_positions = repeated_cells_atom_positions.reset_index()

        # Repeat in the x-dir and y-dir according to n_extend_x and n_extend_y
        repeated_cells_atom_positions = pd.DataFrame(columns=unit_cell_atom_positions.columns)
        for i in range(self.n_extend_x):
            for j in range(self.n_extend_y):
                repeated_cells_atom_positions = pd.concat(
                    [repeated_cells_atom_positions, unit_cell_atom_positions.iloc[0:n_unit_cell_atoms * self.n_layers]],
                    ignore_index=True)
                repeated_cells_atom_positions.iloc[-n_unit_cell_atoms * self.n_layers:].loc[:, ["x", "y", "z"]] = \
                    repeated_cells_atom_positions.iloc[-n_unit_cell_atoms * self.n_layers:][["x", "y", "z"]] + (
                                self.slab_a1 / self.n_extend_x) * i + (self.slab_a2 / self.n_extend_y) * j

        # Reset index
        unit_cell_atom_positions = repeated_cells_atom_positions.reset_index()

        # Replace the atom types of the starting geometry file with the atom types stored in the atom types file
        atom_types_file = self.parent_folder + "atom_types/atom_types_" + self.prefix
        unit_cell_atom_positions.loc[:, "Atom_type"] = pd.read_table(atom_types_file, delim_whitespace=True)[
            "Atom_type"]
        unit_cell_atom_positions = unit_cell_atom_positions.drop(columns=["level_0"])

        # Remove atoms that are marked with "removed" in atom_types, reset index and define "ID", that is index+1
        unit_cell_atom_positions = unit_cell_atom_positions[unit_cell_atom_positions["Atom_type"] != "removed"]
        unit_cell_atom_positions.reset_index(inplace=True)
        unit_cell_atom_positions.loc[:, "ID"] = unit_cell_atom_positions.index + 1
        unit_cell_atom_positions = unit_cell_atom_positions.drop(columns=["level_0"])

        if return_unit_cell_atom_positions:
            return unit_cell_atom_positions

        try:
            # Check if hopping matrix already exists with this file name to avoid overwriting
            output_file = open(
                self.parent_folder + "hopping_matrix/hopping_matrix_" + self.prefix, "r")
            print("Hopping matrix already exists")

        except:
            # Repeat full unit cell once in x-dir and y-dir because the neighbouring atoms must be there to identify the hopping terms out of the unit cell
            repeated_cells_atom_positions = pd.DataFrame(columns=unit_cell_atom_positions.columns)
            for i in [-1, 0, 1]:
                for j in [-1, 0, 1]:
                    repeated_cells_atom_positions = pd.concat(
                        [repeated_cells_atom_positions, unit_cell_atom_positions.iloc[0:self.n_atoms]],
                        ignore_index=True)
                    repeated_cells_atom_positions.iloc[-self.n_atoms:].loc[:, ["x", "y", "z"]] = \
                    repeated_cells_atom_positions.iloc[-self.n_atoms:][
                        ["x", "y", "z"]] + self.slab_a1 * i + self.slab_a2 * j

            # Define vectors connecting 1st NN
            vectors_1st_nn_1 = self.a1 - self.d
            vectors_1st_nn_2 = self.a2 - self.d
            vectors_1st_nn_3 = self.a3 - self.d
            vectors_1st_nn = np.array([vectors_1st_nn_1, vectors_1st_nn_2, vectors_1st_nn_3])

            # Define vectors connecting 2nd NN
            vectors_2nd_nn_1 = self.a1 + self.a3 - self.d
            vectors_2nd_nn_2 = self.a2 + self.a3 - self.d
            vectors_2nd_nn_3 = self.a2 + self.a1 - self.d
            vectors_2nd_nn = np.array([vectors_2nd_nn_1, vectors_2nd_nn_2, vectors_2nd_nn_3])

            # Define vectors connecting 3rd NN
            vectors_3rd_nn_1 = self.a1 - self.a3
            vectors_3rd_nn_2 = self.a2 - self.a3
            vectors_3rd_nn_3 = self.a1 - self.a2
            vectors_3rd_nn = np.array([vectors_3rd_nn_1, vectors_3rd_nn_2, vectors_3rd_nn_3])

            # Define 1st, 2nd and 3rd NN distances, and the distance error margin (add_nn_dist)
            add_nn_dist = 0.1000
            dist_3rd_nn = np.linalg.norm(vectors_3rd_nn_1)
            dist_2nd_nn = np.linalg.norm(vectors_2nd_nn_1)
            dist_1st_nn = np.linalg.norm(vectors_1st_nn_1)

            # Function to get the difference between two vectors
            def get_displacement_vector(v1, v2):
                return np.array(v2[["x", "y", "z"]] - v1[["x", "y", "z"]], dtype=float)

            # Function that returns the index of the Nth NN vector with sign (ie whether it is vector 1,2 or 3 (plus/minus) of vectors_nn that displacement_vec connects)
            def get_nn_vector_index(displacement_vec, vectors_nn):
                index = 1
                for vec in vectors_nn:
                    if np.isclose(vec, displacement_vec, rtol=5e-3, atol=5e-3).all():
                        return -index
                    elif np.isclose(-vec, displacement_vec, rtol=5e-3, atol=5e-3).all():
                        return index
                    else:
                        index += 1
                print("Error: Could not find vector connecting atoms!")
                return None

            # Write hopping matrix to this file
            output_file = open(
                self.parent_folder + "hopping_matrix/hopping_matrix_" + self.prefix, "w")

            # Loop through all atoms in the unit cell
            for index_unit_cell, atom_unit_cell in unit_cell_atom_positions.iterrows():

                # For each atom in the unit cell, loop through each atom in the repeated unit cell
                for index_repeated_cell, atom_repeated_cell in repeated_cells_atom_positions.iterrows():

                    # Get displacement vector (and distance) between uc atom and repeated uc atom
                    displacement_vec = get_displacement_vector(atom_unit_cell, atom_repeated_cell)
                    dist = np.linalg.norm(displacement_vec)

                    # Do nothing if dist is more than 3rd NN
                    if dist > dist_3rd_nn + add_nn_dist:
                        continue

                    # else if dist is more than 2nd NN, write 3rd NN term
                    elif dist > dist_2nd_nn + add_nn_dist:
                        atom_types = " " + atom_unit_cell["Atom_type"] + " " + atom_repeated_cell["Atom_type"]
                        output_file.write(str(atom_unit_cell["ID"]) + ' ' + str(atom_repeated_cell["ID"]) + ' 3 ' + str(
                            get_nn_vector_index(displacement_vec, vectors_3rd_nn)) + atom_types + '\n')

                    # else if dist is more than 1st NN, write 2nd NN term
                    elif dist > dist_1st_nn + add_nn_dist:
                        atom_types = " " + atom_unit_cell["Atom_type"] + " " + atom_repeated_cell["Atom_type"]
                        output_file.write(str(atom_unit_cell["ID"]) + ' ' + str(atom_repeated_cell["ID"]) + ' 2 ' + str(
                            get_nn_vector_index(displacement_vec, vectors_2nd_nn)) + atom_types + '\n')

                    # else if dist is more than onsite, write 1st NN term
                    elif dist > add_nn_dist:
                        atom_types = " " + atom_unit_cell["Atom_type"] + " " + atom_repeated_cell["Atom_type"]
                        output_file.write(str(atom_unit_cell["ID"]) + ' ' + str(atom_repeated_cell["ID"]) + ' 1 ' + str(
                            get_nn_vector_index(displacement_vec, vectors_1st_nn)) + atom_types + '\n')

                    # else write onsite term
                    else:
                        atom_types = " " + atom_unit_cell["Atom_type"] + " " + atom_repeated_cell["Atom_type"]
                        output_file.write(
                            str(atom_unit_cell["ID"]) + ' ' + str(atom_repeated_cell["ID"]) + ' 0' + atom_types + '\n')

                output_file.write('\n')

            output_file.close()
            print("Hopping matrix successfully written to file")

    def write_xsf(self, filename, atom_positions_file, crystal_c_v3):
        # Write the slab configuration to a xsf file that can be opened with xcrysden

        unit_cell_atom_positions = self.define_hopping_matrix(atom_positions_file, crystal_c_v3,
                                                              return_unit_cell_atom_positions=True)

        output_file = open(self.parent_folder + "xsf/" + filename, "w")
        output_file.write(" CRYSTAL \n\n PRIMVEC\n")

        for vec in [self.slab_a1, self.slab_a2, self.slab_a3]:
            output_file.write('\t')
            output_file.write(np.array2string(vec).replace("[", "").replace("]", ""))
            output_file.write('\n')

        output_file.write("\n CONVVEC\n")

        for vec in [self.slab_a1, self.slab_a2, self.slab_a3]:
            output_file.write('\t')
            output_file.write(np.array2string(vec).replace("[", "").replace("]", ""))
            output_file.write('\n')

        output_file.write("\n PRIMCOORD\n")
        output_file.write(" " + str(self.n_atoms) + " 1\n")
        output_file.write(
            unit_cell_atom_positions[["Atom_type", "x", "y", "z"]].to_string(header=False, index=False).replace("Bi",
                                                                                                                "83").replace(
                "Sb", "51"))
        output_file.close()
        print("Atoms written to file: " + filename)

    def save_eigvals(self, eigvals, Ef, keyword_savefile):
        # Save eigenvalues to text file

        # Save eigenvalues
        f = open(
            self.parent_folder + "eigenvalues/" + keyword_savefile + "_" + self.prefix, "w")
        f.write("Ef = " + str(Ef) + '\n')
        for i in range(len(eigvals)):
            f.write(np.array2string(eigvals[i], threshold=int(1e9)).replace("[", "").replace("]", "") + '\n')
        f.close()

    def load_eigvals(self, keyword_savefile):
        # Load eigenvalues from text file
        eigvals = []

        # Load eigenvalues
        f = open(
            self.parent_folder + "eigenvalues/" + keyword_savefile + "_" + self.prefix, "r")
        Ef = float(f.readline().split()[-1])
        for line in f:
            line_values = line.split()
            eigvals.extend(line_values)
        f.close()
        eigvals = np.array(eigvals, dtype=float).reshape((len(eigvals) // self.n_bnd, self.n_bnd))
        return eigvals, Ef

    def save_k_points(self, k_points, keyword_savefile):
        # Save k_points to text file
        # Save k_points
        f = open(
            self.parent_folder + "k_points/" + keyword_savefile + "_" + self.prefix, "w")
        for i in range(len(k_points)):
            f.write(np.array2string(k_points[i], threshold=int(1e9)).replace("[", "").replace("]", "") + '\n')
        f.close()

    def load_k_points(self, keyword_savefile):
        # Load k_points from text file
        k_points = []

        # Load eigenvalues
        f = open(
            self.parent_folder + "k_points/" + keyword_savefile + "_" + self.prefix, "r")
        for line in f:
            line_values = line.split()
            k_points.extend(line_values)
        f.close()
        k_points = np.array(k_points, dtype=float).reshape((len(k_points) // 3, 3))

        return k_points

    def save_edge_participation(self, edge_participation, keyword_savefile):
        # Save edge_participation to text file

        # Save edge_participation
        f = open(
            self.parent_folder + "edge_participation/" + keyword_savefile + "_" + self.prefix, "w")
        for i in range(len(edge_participation)):
            f.write(np.array2string(edge_participation[i], threshold=int(1e9)).replace("[", "").replace("]", "") + '\n')
        f.close()

    def load_edge_participation(self, keyword_savefile):
        # Load edge_participation from text file
        edge_participation = []

        # Load edge_participation
        f = open(
            self.parent_folder + "edge_participation/" + keyword_savefile + "_" + self.prefix, "r")
        for line in f:
            line_values = line.split()
            edge_participation.extend(line_values)
        f.close()
        edge_participation = np.array(edge_participation, dtype=float).reshape(
            (len(edge_participation) // self.n_bnd, self.n_bnd))

        return edge_participation

    def save_spin_expectation_values(self, spin_expectation_values, keyword_savefile):
        # Save expecation values to text file

        # Save expectation values
        f = open(
            self.parent_folder + "spin_expectation_values/" + keyword_savefile + "_" + self.prefix, "w")
        f.write("X\tY\tZ" + '\n')
        for i in range(len(spin_expectation_values)):
            f.write(np.array2string(spin_expectation_values[i], threshold=int(1e9)).replace("[", "").replace("]",
                                                                                                             "") + '\n')
        f.close()

    def load_spin_expectation_values(self, keyword_savefile):
        # Load spin_expectation_values from text file
        spin_expectation_values = []

        # Load eigenvalues
        f = open(
            self.parent_folder + "spin_expectation_values/" + keyword_savefile + "_" + self.prefix, "r")
        f.readline()  # X Y Z

        for line in f:
            line_values = line.split()
            spin_expectation_values.extend(line_values)

        f.close()
        spin_expectation_values = np.array(spin_expectation_values, dtype=float).reshape(
            (len(spin_expectation_values) // (self.n_bnd * 3), self.n_bnd, 3))
        return spin_expectation_values

    def calculate_single_eigenvalue(self, i, k_points, hopping_matrix_filename, save_eigvecs,
                                    save_spin_expectation_values, edge_participation_one_sided, select_n_layers):
        # Calculate eigenvalues, eigvecs, edge particiation, and spin exp values for a single k-point (i)

        # Select k_point i
        k = k_points[i]

        # Calculate eigvals and eigvecs
        current_eigval, current_eigvec = linalg.eigh(self.get_hamiltonian(k, hopping_matrix_filename))

        # Calculate edge participation and spin exp values
        if save_spin_expectation_values or edge_participation_one_sided:
            edge_participation = self.get_edge_participation_one_sided(current_eigvec, select_n_layers)
        else:
            edge_participation = self.get_edge_participation(current_eigvec)

        if save_spin_expectation_values:
            spin_expectation_value = self.get_spin_expectation_values(current_eigvec)
        else:
            spin_expectation_value = 0

        """
        if save_eigvecs:
            # Save eigenvectors
            # (!) This function does not work atm because I implemented parallel execution. If you want to save eigvecs, you have to turn off parallel execution and do it outside this function!
            f_eigvecs.write(np.array2string(k).replace("[", "").replace("]", ""))
            f_eigvecs.write(
                '\n' + np.array2string(current_eigvec, threshold=int(1e9)).replace("[", "").replace("]", "") + '\n')
        """

        return current_eigval, edge_participation, spin_expectation_value

    def calculate_eigenvalues(self, k_points, keyword_savefile, save_eigvecs=False, save_spin_expectation_values=False,
                              edge_participation_one_sided=False, select_n_layers=5):
        # Diagonalize the Hamiltonian defined as a hopping matrix, store eigenvalues and eigenvectors

        hopping_matrix_filename = "hopping_matrix/hopping_matrix_" + self.prefix

        try:
            output_file = open(self.parent_folder + hopping_matrix_filename, "r")
            output_file.close()
            print("Existing hopping matrix found, will be used\n")
        except:
            print("No existing hopping matrix found, please define hopping matrix before calculating band structure")
            return None

        if save_eigvecs:
            # Save eigenvectors
            f_eigvecs = open(
                self.parent_folder + "eigenvectors/" + keyword_savefile + "_" + self.prefix, "w")

        ### Diagonalize the Hamiltonian along the selected k_points path ###

        # Time it
        secs_tot = time.time()

        # Parallel execution of calculating eigenvalues (and more) for all kpoints, n_jobs=8 around 3-4 times faster than single core execution
        results = Parallel(n_jobs=8)(
            delayed(self.calculate_single_eigenvalue)(i, k_points, hopping_matrix_filename, save_eigvecs,
                                                      save_spin_expectation_values, edge_participation_one_sided,
                                                      select_n_layers) for i in range(len(k_points)))

        # Initialize empty lists for results
        eigvals = []
        edge_participation = []
        spin_expectation_values = []

        # Store results in the lists
        for i in range(len(k_points)):
            eigvals.append(results[i][0])
            edge_participation.append(results[i][1])
            spin_expectation_values.append(results[i][2])

        if save_eigvecs:
            f_eigvecs.close()
            print("\nSaved eigenvectors to file\n")
        if save_spin_expectation_values:
            self.save_spin_expectation_values(spin_expectation_values, keyword_savefile)
            print("\nSaved spin expectation values to file\n")

        eigvals = np.array(eigvals)
        edge_participation = np.array(edge_participation)
        print("generated and solved all matrices in " + str((time.time() - secs_tot) / 60) + " minutes")

        # Fill up bands to get HOMO, LUMO and Ef
        n_filling = len(k_points)
        HOMO = np.sort(np.reshape(eigvals, n_filling * self.n_bnd))[
            n_filling * self.n_atoms * self.electrons_per_atom - 1]
        LUMO = np.sort(np.reshape(eigvals, n_filling * self.n_bnd))[n_filling * self.n_atoms * self.electrons_per_atom]
        Ef = (HOMO + LUMO) / 2

        print("\nFermi level: " + str(round(Ef, 3)) + "eV")

        # Save eigenvalues and edge participation
        self.save_k_points(k_points, keyword_savefile)
        self.save_eigvals(eigvals, Ef, keyword_savefile)
        self.save_edge_participation(edge_participation, keyword_savefile)
        print("\nSaved k-points, eigenvalues and edge participation to file")

    def calculate_band_structure(self, symmetry_points_list, resolution, keyword_savefile="band_structure",
                                 save_eigvecs=False, save_spin_expectation_values=False,
                                 edge_participation_one_sided=False, select_n_layers=5):
        # Calculate band structure along the defined high symmetry points

        # Make a list of kpoints between high symmetry points
        k_points = np.empty((0, 3))
        for point in range(len(symmetry_points_list) - 1):
            k_points = np.concatenate(
                (k_points, np.linspace(symmetry_points_list[point], symmetry_points_list[point + 1], resolution)))

        # Eigenvalues and eigenvectors calulated and saved to file
        self.calculate_eigenvalues(k_points, keyword_savefile, save_eigvecs, save_spin_expectation_values,
                                   edge_participation_one_sided, select_n_layers)

    def calculate_constant_energy_surfaces(self, k_space_edges_list, resolution,
                                           keyword_savefile="constant_energy_surfaces", save_eigvecs=False,
                                           save_spin_expectation_values=False, edge_participation_one_sided=False,
                                           select_n_layers=5, mirror_symmetry=True):
        # Calculate band structure along the defined high symmetry points, k_space_edges_list = [Gamma, X1, X2],
        # resolution is along Gamma-X1
        # X1 and X2 must be normal, or the plot will be skewed!

        # Make a grid of kpoints in the defined region of k-space
        k_points = np.empty((0, 3))
        Gamma = np.zeros(3, dtype=float)
        X1 = k_space_edges_list[0]
        X2 = k_space_edges_list[1]
        resolution_X2 = int(resolution * np.linalg.norm(X2) / np.linalg.norm(X1))
        shift = X2 / resolution_X2

        if mirror_symmetry:
            for point in range(resolution_X2 // 2 + 1):
                k_points = np.concatenate(
                    (k_points, np.linspace(Gamma + shift * point * 2, X1 + shift * point * 2, resolution // 2 + 1)))
        else:
            k_points = np.concatenate((k_points, np.linspace(-X1, X1, resolution + 1)))
            for point in range(1, resolution_X2 // 2 + 1):
                k_points = np.concatenate(
                    (k_points, np.linspace(-X1 + shift * point * 2, X1 + shift * point * 2, resolution + 1)))
                k_points = np.concatenate(
                    (k_points, np.linspace(-X1 - shift * point * 2, X1 - shift * point * 2, resolution + 1)))

        # Eigenvalues and eigenvectors calulated and saved to file
        self.calculate_eigenvalues(k_points, keyword_savefile, save_eigvecs, save_spin_expectation_values,
                                   edge_participation_one_sided, select_n_layers)

    def plot_band_structure(self, symmetry_points_list, keyword_savefile="band_structure", load_eigvecs=False,
                            labels=["Gamma", "X2", "M", "Gamma", "X1", "M"],
                            save_plot=False, E_min=-1.05, E_max=0.35, set_Ef=None, load_spin_expectation_values=False,
                            spin_comp=2, use_edge_participation=True):
        ### Plot the already calculated band structure along high symmetry path ###

        # Load kpoints and eigenvalues (including edge participation and spin)
        k_points = self.load_k_points(keyword_savefile)
        eigvals, Ef = self.load_eigvals(keyword_savefile)
        edge_participation = self.load_edge_participation(keyword_savefile)
        resolution = len(k_points) // (len(symmetry_points_list) - 1)
        if load_spin_expectation_values:
            spin_expectation_values = self.load_spin_expectation_values(keyword_savefile)[:, :, spin_comp]

        # Overwrite Ef if wanted
        if set_Ef != None:
            Ef = set_Ef

        # Make a k_path list that holds the distance travelled along the k_path
        k_path = [0]
        for k_nr in range(1, len(k_points)):
            k_path.append(k_path[-1] + np.linalg.norm(k_points[k_nr] - k_points[k_nr - 1]))

        if load_eigvecs:
            # Load eigenvectors
            f_eigvecs = open(
                self.parent_folder + "eigenvectors/" + keyword_savefile + "_" + self.prefix, "r")

            for i in range(len(k_points)):
                current_eigvec = []
                f_eigvecs.readline()
                for j in range(self.n_bnd ** 2 // 2):
                    current_eigvec.extend(f_eigvecs.readline().split())
                current_eigvec = np.array(current_eigvec, dtype=complex).reshape((self.n_bnd, self.n_bnd))
                edge_participation.append(self.get_edge_participation(current_eigvec))
            edge_participation = np.array(edge_participation)

        # Plot band structure along entire high symmetry path
        fig = plt.figure(figsize=(12, 10))

        # Indices of the x-axis
        old_indices = np.arange(0, len(k_path))

        # Increased reolution of the x-axis (times 10) using spline on the k_path
        new_length = len(k_path) * 10
        new_indices = np.linspace(0, len(k_path) - 1, new_length)
        spl = UnivariateSpline(old_indices, k_path, k=1, s=0)
        k_path_dense = spl(new_indices)

        # Define color map for spin
        cm1 = mcol.LinearSegmentedColormap.from_list("MyCmapName", ["b", "r"])

        # Loop through bands
        for bnd in range(self.n_bnd):

            # Use only eigvals in the plotting range (energy min-max)
            if (eigvals[:, bnd] - Ef > E_min).any() and (eigvals[:, bnd] - Ef < E_max).any():

                # Use spline to increase resolution of eigvals and edge participation, same as for k_path
                spl = UnivariateSpline(old_indices, eigvals[:, bnd], k=1, s=0)
                eigvals_dense = spl(new_indices)
                spl = UnivariateSpline(old_indices, edge_participation[:, bnd], k=1, s=0)
                edge_participation_dense = spl(new_indices)

                if load_spin_expectation_values:
                    # If using spin_exp_values, the spline to increase resolution is NOT IMPLEMENTED, use instead normal resolution
                    # spl = UnivariateSpline(old_indices, spin_expectation_values[:, bnd], k=1, s=0)
                    # spin_expectation_values_dense = spl(new_indices)

                    # Plot only states on one side, ie where edge participation is large enough (using one-sided edge participation)
                    idx_where_edge_states = np.argwhere(edge_participation[:, bnd] > 0.25)
                    plt.scatter(np.array(k_path)[idx_where_edge_states], eigvals[:, bnd][idx_where_edge_states] - Ef,
                                c=spin_expectation_values[:, bnd][idx_where_edge_states], cmap=cm1, s=8, marker="o",
                                rasterized=True, vmin=-1, vmax=1)
                elif use_edge_participation:
                    # Plot with gray cmap if using edge participation without spin
                    plt.scatter(k_path_dense, eigvals_dense - Ef, c=edge_participation_dense, cmap="gray_r", s=6,
                                marker=".", rasterized=True)
                else:
                    # Plot all in blue if not using spin or edge participation
                    plt.scatter(k_path_dense, eigvals_dense - Ef, c="tab:blue", s=6, marker=".", rasterized=True)

        # Add vertical lines for high symmetry points and horizontal line at Fermi level
        plt.vlines([*k_path[::resolution], k_path[-1]], ymin=np.min(eigvals), ymax=np.max(eigvals), linestyles="--",
                   colors="k")
        plt.hlines(0, xmin=0, xmax=np.max(k_path), linestyles="dotted", colors="b")

        # Add high symmetry points labels
        plt.xticks([*k_path[::resolution], k_path[-1]], labels=labels)

        # Set energy scale limits
        plt.ylim(E_min, E_max)
        plt.tight_layout()

        if save_plot:
            plt.savefig(
                self.parent_folder[:-19] + "plots/" + keyword_savefile + "_" + self.prefix + ".pdf", format="pdf",
                bbox_inches='tight')
        else:
            plt.show()

    def plot_constant_energy_surfaces(self, k_space_edges_list, resolution, keyword_savefile="constant_energy_surfaces",
                                      load_eigvecs=False, save_plot=False,
                                      sigma=0.05, Eb_list=[0.0], temperature=298, weight_edge_participation=True,
                                      set_Ef=None, mirror_symmetry=True):
        ### Plot the already calculated constant energy surfaces ###

        # Load kpoints and eigenvalues (including edge participation and spin)
        k_points = self.load_k_points(keyword_savefile)
        eigvals, Ef = self.load_eigvals(keyword_savefile)
        edge_participation = self.load_edge_participation(keyword_savefile)

        # Overwrite Ef if wanted
        if set_Ef != None:
            Ef = set_Ef

        # Set high symmetry points based on list input
        X1 = k_space_edges_list[0]
        X2 = k_space_edges_list[1]
        resolution_X2 = int(resolution * np.linalg.norm(X2) / np.linalg.norm(X1))

        if load_eigvecs:
            # Load eigenvectors
            f_eigvecs = open(
                self.parent_folder + "eigenvectors/" + keyword_savefile + "_" + self.prefix, "r")

            for i in range(len(k_points)):
                current_eigvec = []
                f_eigvecs.readline()
                for j in range(self.n_bnd ** 2 // 2):
                    current_eigvec.extend(f_eigvecs.readline().split())
                current_eigvec = np.array(current_eigvec, dtype=complex).reshape((self.n_bnd, self.n_bnd))
                edge_participation.append(self.get_edge_participation(current_eigvec))
            edge_participation = np.array(edge_participation)

        # Energy threshold for including a kpoint in the heatmap_matrix
        boltzmann_energy = 8.62 * 1e-5 * temperature  # Boltzmann constant * temperature

        plot_shape = Eb_list.shape
        fig, axis = plt.subplots(plot_shape[0], plot_shape[1], figsize=np.array(plot_shape) * 6)
        # Loop through all calculated kpoints for all Eb and all of their eigenvalues
        count = 0
        for Eb in Eb_list.reshape(Eb_list.size):

            # List with all kpoints that have an eigenvalue equal Ef - Eb
            heatmap_matrix = []

            for kpoint_nr in range(len(k_points)):
                kpoint = k_points[kpoint_nr]

                kx = np.dot(X1, kpoint)
                ky = np.dot(X2, kpoint)

                fermi_dist = 1 / (np.exp((eigvals[kpoint_nr, :] - Ef) / boltzmann_energy) + 1)
                weight = fermi_dist * np.exp((-(eigvals[kpoint_nr, :] - (Ef - Eb)) ** 2) / (2 * sigma ** 2))
                if weight_edge_participation:
                    weight *= edge_participation[kpoint_nr, :]
                weight = np.sum(weight)

                heatmap_matrix.append([kx, ky, weight])

                if ky > 1e-8 and mirror_symmetry:
                    heatmap_matrix.append([kx, -ky, weight])
                if kx > 1e-8 and mirror_symmetry:
                    heatmap_matrix.append([-kx, ky, weight])
                if kx > 1e-8 and ky > 1e-8 and mirror_symmetry:
                    heatmap_matrix.append([-kx, -ky, weight])

            heatmap_matrix = np.array(heatmap_matrix)
            heatmap, xedges, yedges = np.histogram2d(heatmap_matrix[:, 1], heatmap_matrix[:, 0],
                                                     bins=(resolution_X2, resolution + 1), weights=heatmap_matrix[:, 2])
            extent = [-np.linalg.norm(X1) / 2, np.linalg.norm(X1) / 2, -np.linalg.norm(X2) / 2, np.linalg.norm(X2) / 2]

            ax = axis.reshape(axis.size)[count]
            count += 1

            ax.set_title("E$_{b}$ = " + str(round(Eb, 1)), fontsize=11)
            ax.set_xlabel('Å$^{-1}$', fontsize=11)
            ax.set_ylabel('Å$^{-1}$', fontsize=11)
            ax.tick_params(axis='both', which='major', labelsize=11, direction='out', length=4, width=1.5)

            ax.locator_params(nbins=6, axis='x')
            ax.locator_params(nbins=6, axis='y')

            ax_r = ax.secondary_yaxis('right')
            ax_t = ax.secondary_xaxis('top')
            ax_r.tick_params(axis='y', direction='in')
            ax_t.tick_params(axis='x', direction='in')
            ax_t.locator_params(nbins=6, axis='x')
            ax_r.locator_params(nbins=6, axis='y')
            ax_t.tick_params(axis='both', which='major', labelsize=0, direction='out', length=4, width=1.5)
            ax_r.tick_params(axis='both', which='major', labelsize=0, direction='out', length=4, width=1.5)

            ax.imshow(heatmap, cmap='gray_r', extent=extent)
            make_axes_locatable(ax)

        fig.tight_layout()

        if save_plot:
            plt.savefig(
                self.parent_folder[:-19] + "plots/" + keyword_savefile + "_" + self.prefix + ".pdf", format="pdf",
                bbox_inches='tight')
        else:
            plt.show()


class ModelBulk(ModelCrystal):

    def __init__(self, prefix, atom_types, n_extend, kwargs=None):
        """
        Defines a Bi, Sb or BiSb (surface_normal e.g. 112 or 111)-slab that is produced by repeating the Liu-Allen geometry (lattice_const_a/c) of 4 atoms n_layers times
        in the z-direction and n_extend_x/y in the x/y-directions. n_layers should be an even number because of how get_edge_participation is
        implemented. n_atoms_removed is for removing spcific atoms from the generated crystal to generate alternative surfaces (truncated bulk
        if n_atoms_removed=0, reconstruction I (II) with 1 (2) atoms removed). atom_types is for defining the crystal geomtry (either Bi or Sb
        from Liu-Allen) and for choosing the working folder for storing/reading files. Supported values of slab_type is currently {truncated_bulk,
        truncated_bulk_BiSb, crystal17 (aka lines with stochiometry Bi13Sb3), reconstruction_typeI, reconstruction_typeII}. kwargs is for defining
        the BiSb hopping params from those of Bi and Sb (factor_1, scale_Bi_Sb = kwargs). After initializing the model, one typically runs
        define_atom_types to generate a list with atom numbers and atom types, it is stored in the atom_types folder. Then one can run
        define_hopping_matrix to generate a list containing all atom numbers, all their neighbouring atom numbers, two numbers that descibe 1st, 2nd
        or 3rd neighbour hopping plus the hopping vector (a1,a2 or a3), and the atom types (Bi or Sb). The hopping matrix is stored as a text file
        in the hopping_matrix_folder. Then one can run calculate_bandstructure along a high symmetry path or calculate_constant_energy_surface to
        solve the hopping matrices for certain values of k, see details in the functions. Lastly one can plot the band structure using plot_band_structure
        or plot_constant_energy_surface, choosing parameters for plotting such as spin exp values or edge participation.
        """

        #############################################################################
        ### Define the crystal lattice parameters and the hopping matrix elements ###
        #############################################################################

        # Class definition parameters
        super().__init__(atom_types, prefix)
        self.n_extend = n_extend

        # Total number of atoms in the unit cell
        self.n_atoms = 2 * self.n_extend[0] * self.n_extend[1] * self.n_extend[2]

        # Number of localized atomic orbitals per atom (including spin)
        self.n_bnd = self.n_orbitals * self.n_atoms

        # Extended lattice vectors
        self.a1_ext = self.n_extend[0] * self.a1
        self.a2_ext = self.n_extend[1] * self.a2
        self.a3_ext = self.n_extend[2] * self.a3

        # Slab reciprocal lattice vectors
        V = np.dot(self.a1_ext, np.cross(self.a2_ext, self.a3_ext))
        self.b1_ext = (2 * np.pi / V) * np.cross(self.a2_ext, self.a3_ext)
        self.b2_ext = (2 * np.pi / V) * np.cross(self.a3_ext, self.a1_ext)
        self.b3_ext = (2 * np.pi / V) * np.cross(self.a1_ext, self.a2_ext)
        self.b_lattice_vectors_ext = np.array([self.b1_ext, self.b2_ext, self.b3_ext])

    def get_hamiltonian(self, k, file_name_hopping_matrix):
        ### Define all the individual hopping matrices "H_n_n" of the upper triangular part of H ###

        # Get single-site hopping matrices
        H_all_nn_Bi = self.get_H_all_nn_Bi()
        H_all_nn_Sb = self.get_H_all_nn_Sb()
        H_all_nn_Bi_Sb = self.get_H_all_nn_Bi_Sb()

        # Initiate zero matrix
        H = np.zeros((self.n_bnd, self.n_bnd), dtype=complex)

        # Read hopping matrix from file
        hopping_matrix = open(self.parent_folder + file_name_hopping_matrix, 'r')

        # Loop through hopping terms in the matrix
        for hopping_element_str in hopping_matrix:

            # Skip empty lines
            if hopping_element_str == '\n':
                continue

                # Read line in hopping matrix, extract atom numbers and types
            hopping_element = np.array(hopping_element_str[:-7].split(' '), dtype=int)
            atom1 = hopping_element[0] - 1
            atom2 = hopping_element[1] - 1
            atom_type1 = hopping_element_str[-6:-4]
            atom_type2 = hopping_element_str[-3:-1]

            # Check if hopping is Bi-Bi, Sb-Sb, or Bi-Sb
            if atom_type1 == "Bi" and atom_type2 == "Bi":
                if hopping_element[2] == 0:
                    # Onsite hopping
                    element = self.get_H_onsite_Bi()
                else:
                    # Sign determines direction of hopping, then choose the correct single-site hopping term (e.g. 1st-NN-2)
                    element_sign = np.sign(hopping_element[3])
                    if element_sign == 1:
                        element = H_all_nn_Bi[hopping_element[2] - 1][abs(hopping_element[3]) - 1](k)
                    else:
                        element = np.transpose(
                            np.conjugate(H_all_nn_Bi[hopping_element[2] - 1][abs(hopping_element[3]) - 1](k)))

            elif atom_type1 == "Sb" and atom_type2 == "Sb":
                if hopping_element[2] == 0:
                    # Onsite hopping
                    element = self.get_H_onsite_Sb()
                else:
                    # Sign determines direction of hopping, then choose the correct single-site hopping term (e.g. 1st-NN-2)
                    element_sign = np.sign(hopping_element[3])
                    if element_sign == 1:
                        element = H_all_nn_Sb[hopping_element[2] - 1][abs(hopping_element[3]) - 1](k)
                    else:
                        element = np.transpose(
                            np.conjugate(H_all_nn_Sb[hopping_element[2] - 1][abs(hopping_element[3]) - 1](k)))


            elif atom_type1 == "Bi" and atom_type2 == "Sb" or atom_type1 == "Sb" and atom_type2 == "Bi":
                if hopping_element[2] == 0:
                    # Onsite hopping Bi-Sb not possible
                    raise Exception("Onsite hopping between Bi and Sb")
                else:
                    # Sign determines direction of hopping, then choose the correct single-site hopping term (e.g. 1st-NN-2)
                    element_sign = np.sign(hopping_element[3])
                    if element_sign == 1:
                        element = H_all_nn_Bi_Sb[hopping_element[2] - 1][abs(hopping_element[3]) - 1](k)
                    else:
                        element = np.transpose(
                            np.conjugate(H_all_nn_Bi_Sb[hopping_element[2] - 1][abs(hopping_element[3]) - 1](k)))

            else:
                raise Exception("Atom type not Bi or Sb")

            # Add the hopping term for this element in the hopping matrix loop
            H[atom1 * self.n_orbitals:(atom1 + 1) * self.n_orbitals, atom2 * self.n_orbitals:(atom2 + 1) * self.n_orbitals] += element

        # Close text file
        hopping_matrix.close()

        # Throw Exception if H is not Hermitian
        if not np.isclose(np.transpose(np.conjugate(H)), H).all():
            # print(np.where(np.not_equal(np.isclose(np.transpose(np.conjugate(H)), H), True)))
            raise Exception("H is not hermitian")

        return H

    def define_atom_types(self):
        # Available type: pure bulk

        # Write pure truncated Bi or Sb bulk atom types to file
        file = open(
            self.parent_folder + "atom_types/atom_types_" + self.prefix, "w")
        file.write("ID Atom_type\n")
        for atom_nr in range(self.n_atoms):
            file.write(str(atom_nr + 1) + " " + "Bi" + "\n")
        file.close()

        print("Atoms types successfully written to file")
        return None

    def define_hopping_matrix(self, atom_positions_file, return_unit_cell_atom_positions=False):
        ### Generate a text file containing all the atoms with types, and all neighbours (1st-3rd) ###

        # Read starting geometry (n atoms that will be repeated to create slab)
        unit_cell_atom_positions = pd.read_csv(
            self.parent_folder + "atom_positions/" + atom_positions_file, delim_whitespace=True)
        n_unit_cell_atoms = len(unit_cell_atom_positions)

        # Repeat in the x-dir, y-dir and z-dir according to n_extend
        repeated_cells_atom_positions = pd.DataFrame(columns=unit_cell_atom_positions.columns)
        for i in range(self.n_extend[0]):
            for j in range(self.n_extend[1]):
                for k in range(self.n_extend[2]):
                    repeated_cells_atom_positions = pd.concat([repeated_cells_atom_positions, unit_cell_atom_positions.iloc[:n_unit_cell_atoms]], ignore_index=True)
                    repeated_cells_atom_positions.iloc[-n_unit_cell_atoms:].loc[:, ["x", "y", "z"]] = repeated_cells_atom_positions.iloc[-n_unit_cell_atoms:][["x", "y", "z"]] + self.a1 * i + self.a2 * j + self.a3 * k

        # Reset index
        unit_cell_atom_positions = repeated_cells_atom_positions.reset_index()

        # Replace the atom types of the starting geometry file with the atom types stored in the atom types file
        atom_types_file = self.parent_folder + "atom_types/atom_types_" + self.prefix
        unit_cell_atom_positions.loc[:, "Atom_type"] = pd.read_table(atom_types_file, delim_whitespace=True)["Atom_type"]

        # Remove atoms that are marked with "removed" in atom_types, reset index and define "ID", that is index+1
        unit_cell_atom_positions = unit_cell_atom_positions[unit_cell_atom_positions["Atom_type"] != "removed"]
        unit_cell_atom_positions.reset_index(inplace=True)
        unit_cell_atom_positions.loc[:, "ID"] = unit_cell_atom_positions.index + 1
        unit_cell_atom_positions = unit_cell_atom_positions.drop(columns=["level_0"])

        if return_unit_cell_atom_positions:
            return unit_cell_atom_positions

        try:
            # Check if hopping matrix already exists with this file name to avoid overwriting
            output_file = open(
                self.parent_folder + "hopping_matrix/hopping_matrix_" + self.prefix, "r")
            print("Hopping matrix already exists")

        except:
            # Repeat full unit cell once in x-dir, y-dir and z-dir because the neighbouring atoms must be there to identify the hopping terms out of the unit cell
            repeated_cells_atom_positions = pd.DataFrame(columns=unit_cell_atom_positions.columns)
            for i in [-1, 0, 1]:
                for j in [-1, 0, 1]:
                    for k in [-1, 0, 1]:
                        repeated_cells_atom_positions = pd.concat([repeated_cells_atom_positions, unit_cell_atom_positions.iloc[:self.n_atoms]], ignore_index=True)
                        repeated_cells_atom_positions.iloc[-self.n_atoms:].loc[:, ["x", "y", "z"]] = repeated_cells_atom_positions.iloc[-self.n_atoms:][["x", "y", "z"]] + self.a1_ext * i + self.a2_ext * j + self.a3_ext * k

            # Define vectors connecting 1st NN
            vectors_1st_nn_1 = self.a1 - self.d
            vectors_1st_nn_2 = self.a2 - self.d
            vectors_1st_nn_3 = self.a3 - self.d
            vectors_1st_nn = np.array([vectors_1st_nn_1, vectors_1st_nn_2, vectors_1st_nn_3])

            # Define vectors connecting 2nd NN
            vectors_2nd_nn_1 = self.a1 + self.a3 - self.d
            vectors_2nd_nn_2 = self.a2 + self.a3 - self.d
            vectors_2nd_nn_3 = self.a2 + self.a1 - self.d
            vectors_2nd_nn = np.array([vectors_2nd_nn_1, vectors_2nd_nn_2, vectors_2nd_nn_3])

            # Define vectors connecting 3rd NN
            vectors_3rd_nn_1 = self.a1 - self.a3
            vectors_3rd_nn_2 = self.a2 - self.a3
            vectors_3rd_nn_3 = self.a1 - self.a2
            vectors_3rd_nn = np.array([vectors_3rd_nn_1, vectors_3rd_nn_2, vectors_3rd_nn_3])

            # Define 1st, 2nd and 3rd NN distances, and the distance error margin (add_nn_dist)
            add_nn_dist = 0.1000
            dist_3rd_nn = np.linalg.norm(vectors_3rd_nn_1)
            dist_2nd_nn = np.linalg.norm(vectors_2nd_nn_1)
            dist_1st_nn = np.linalg.norm(vectors_1st_nn_1)

            # Function to get the difference between two vectors
            def get_displacement_vector(v1, v2):
                return np.array(v2[["x", "y", "z"]] - v1[["x", "y", "z"]], dtype=float)

            # Function that returns the index of the Nth NN vector with sign (ie whether it is vector 1,2 or 3 (plus/minus) of vectors_nn that displacement_vec connects)
            def get_nn_vector_index(displacement_vec, vectors_nn):
                index = 1
                for vec in vectors_nn:
                    if np.isclose(vec, displacement_vec, rtol=5e-3, atol=5e-3).all():
                        return -index
                    elif np.isclose(-vec, displacement_vec, rtol=5e-3, atol=5e-3).all():
                        return index
                    else:
                        index += 1
                print("Error: Could not find vector connecting atoms!")
                return None

            # Write hopping matrix to this file
            output_file = open(
                self.parent_folder + "hopping_matrix/hopping_matrix_" + self.prefix, "w")

            # Loop through all atoms in the unit cell
            for index_unit_cell, atom_unit_cell in unit_cell_atom_positions.iterrows():

                # For each atom in the unit cell, loop through each atom in the repeated unit cell
                for index_repeated_cell, atom_repeated_cell in repeated_cells_atom_positions.iterrows():

                    # Get displacement vector (and distance) between uc atom and repeated uc atom
                    displacement_vec = get_displacement_vector(atom_unit_cell, atom_repeated_cell)
                    dist = np.linalg.norm(displacement_vec)

                    # Do nothing if dist is more than 3rd NN
                    if dist > dist_3rd_nn + add_nn_dist:
                        continue

                    # else if dist is more than 2nd NN, write 3rd NN term
                    elif dist > dist_2nd_nn + add_nn_dist:
                        atom_types = " " + atom_unit_cell["Atom_type"] + " " + atom_repeated_cell["Atom_type"]
                        output_file.write(str(atom_unit_cell["ID"]) + ' ' + str(atom_repeated_cell["ID"]) + ' 3 ' + str(
                            get_nn_vector_index(displacement_vec, vectors_3rd_nn)) + atom_types + '\n')

                    # else if dist is more than 1st NN, write 2nd NN term
                    elif dist > dist_1st_nn + add_nn_dist:
                        atom_types = " " + atom_unit_cell["Atom_type"] + " " + atom_repeated_cell["Atom_type"]
                        output_file.write(str(atom_unit_cell["ID"]) + ' ' + str(atom_repeated_cell["ID"]) + ' 2 ' + str(
                            get_nn_vector_index(displacement_vec, vectors_2nd_nn)) + atom_types + '\n')

                    # else if dist is more than onsite, write 1st NN term
                    elif dist > add_nn_dist:
                        atom_types = " " + atom_unit_cell["Atom_type"] + " " + atom_repeated_cell["Atom_type"]
                        output_file.write(str(atom_unit_cell["ID"]) + ' ' + str(atom_repeated_cell["ID"]) + ' 1 ' + str(
                            get_nn_vector_index(displacement_vec, vectors_1st_nn)) + atom_types + '\n')

                    # else write onsite term
                    else:
                        atom_types = " " + atom_unit_cell["Atom_type"] + " " + atom_repeated_cell["Atom_type"]
                        output_file.write(
                            str(atom_unit_cell["ID"]) + ' ' + str(atom_repeated_cell["ID"]) + ' 0' + atom_types + '\n')

                output_file.write('\n')

            output_file.close()
            print("Hopping matrix successfully written to file")

    def dist(self, i, j, unit_cell_atom_positions):
        v_i = unit_cell_atom_positions.iloc[i][["x", "y", "z"]]
        v_j = unit_cell_atom_positions.iloc[j][["x", "y", "z"]]

        min_dist = 1e6
        for x in [-1, 0, 1]:
            for y in [-1, 0, 1]:
                for z in [-1, 0, 1]:
                    current_dist = np.linalg.norm(v_i - (v_j + self.a1_ext * x + self.a2_ext * y + self.a3_ext * z))
                    if current_dist < min_dist:
                        min_dist = current_dist

        return min_dist

    def monte_carlo_distribution(self, m, atom_positions_file, crystal_c_v3, step_list, temperature_list):
        """
        Uses a Monte Carlo algorithm to find the subset of m points out of all lattice points
        that maximizes the average pairwise distance.

        Parameters:
        - m: int, number of points to select (ie Sb atoms)
        - dist: function, a function dist(i, j) that gives the distance between points i and j.

        Returns:
        - best_subset: list of int, indices of the points that maximize the average distance.
        - max_avg_min_distance: float, the maximum average distance achieved.
        """

        # Get the atom positions used for dist(i, j)
        unit_cell_atom_positions = self.define_hopping_matrix(atom_positions_file, return_unit_cell_atom_positions=True)

        def greedy_initialization():
            subset = [0]  # Start from the first point
            while len(subset) < m:
                # Choose the next point that maximizes the minimum distance to the points in `subset`
                best_next_point = None
                best_min_distance = 0
                for i in range(self.n_atoms):
                    if i in subset:
                        continue
                    # Calculate the minimum distance from point `i` to the current `subset`
                    closest_distance = min(self.dist(i, j, unit_cell_atom_positions) for j in subset if i != j)
                    if closest_distance > best_min_distance:
                        best_min_distance = closest_distance
                        best_next_point = i
                subset.append(best_next_point)
            print("d = " + str(average_min_distance(subset)))
            return subset

        # Function to calculate the average minimum distance in a given subset of points
        def average_min_distance(subset):
            min_distances = []
            for i in subset:
                # Find the closest distance to any other point in the subset
                closest_distance = min(self.dist(i, j, unit_cell_atom_positions) for j in subset if i != j)
                min_distances.append(closest_distance)
            # return min(min_distances)
            return sum(min_distances) / len(min_distances)

        current_subset = greedy_initialization()
        av_min_dist = average_min_distance(current_subset)
        for j in range(len(step_list)):
            count_list = np.array([0, 0])
            for i in range(step_list[j]):

                atom_to_move = random.randint(0, m - 1)
                new_loc = current_subset[0]
                while new_loc in current_subset:
                    new_loc = random.randint(0, self.n_atoms - 1)

                new_subset = current_subset.copy()
                new_subset[atom_to_move] = new_loc
                new_av_min_dist = average_min_distance(new_subset)

                if np.random.exponential() > (av_min_dist - new_av_min_dist) / temperature_list[j]:
                    current_subset = new_subset.copy()
                    av_min_dist = new_av_min_dist
                    count_list[0] += 1
                else:
                    count_list[1] += 1

            print(count_list)
            print(av_min_dist)
            print("  ")

        # Write BiSb alloy truncated bulk atom types to file
        file = open(
            self.parent_folder + "atom_types/atom_types_" + self.prefix, "w")
        file.write("ID Atom_type\n")
        for atom_nr in range(self.n_atoms):
            if atom_nr in current_subset:
                file.write(str(atom_nr + 1) + " " + "Sb" + "\n")
            else:
                file.write(str(atom_nr + 1) + " " + "Bi" + "\n")
        file.close()

        print(current_subset)
        print(average_min_distance(current_subset))

    def write_xsf(self, filename, atom_positions_file):
        # Write the slab configuration to a xsf file that can be opened with xcrysden

        unit_cell_atom_positions = self.define_hopping_matrix(atom_positions_file,
                                                              return_unit_cell_atom_positions=True)

        output_file = open(self.parent_folder + "xsf/" + filename, "w")
        output_file.write(" CRYSTAL \n\n PRIMVEC\n")

        for vec in [self.a1_ext, self.a2_ext, self.a3_ext]:
            output_file.write('\t')
            output_file.write(np.array2string(vec).replace("[", "").replace("]", ""))
            output_file.write('\n')

        output_file.write("\n CONVVEC\n")

        for vec in [self.a1_ext, self.a2_ext, self.a3_ext]:
            output_file.write('\t')
            output_file.write(np.array2string(vec).replace("[", "").replace("]", ""))
            output_file.write('\n')

        output_file.write("\n PRIMCOORD\n")
        output_file.write(" " + str(self.n_atoms) + " 1\n")
        output_file.write(
            unit_cell_atom_positions[["Atom_type", "x", "y", "z"]].to_string(header=False, index=False).replace("Bi",
                                                                                                                "83").replace(
                "Sb", "51"))
        output_file.close()
        print("Atoms written to file: " + filename)

    def save_eigvals(self, eigvals, Ef, keyword_savefile):
        # Save eigenvalues to text file

        # Save eigenvalues
        f = open(
            self.parent_folder + "eigenvalues/" + keyword_savefile + "_" + self.prefix, "w")
        f.write("Ef = " + str(Ef) + '\n')
        for i in range(len(eigvals)):
            f.write(np.array2string(eigvals[i], threshold=int(1e9)).replace("[", "").replace("]", "") + '\n')
        f.close()

    def load_eigvals(self, keyword_savefile):
        # Load eigenvalues from text file
        eigvals = []

        # Load eigenvalues
        f = open(
            self.parent_folder + "eigenvalues/" + keyword_savefile + "_" + self.prefix, "r")
        Ef = float(f.readline().split()[-1])
        for line in f:
            line_values = line.split()
            eigvals.extend(line_values)
        f.close()
        eigvals = np.array(eigvals, dtype=float).reshape((len(eigvals) // self.n_bnd, self.n_bnd))
        return eigvals, Ef

    def save_k_points(self, k_points, keyword_savefile):
        # Save k_points to text file
        # Save k_points
        f = open(
            self.parent_folder + "k_points/" + keyword_savefile + "_" + self.prefix, "w")
        for i in range(len(k_points)):
            f.write(np.array2string(k_points[i], threshold=int(1e9)).replace("[", "").replace("]", "") + '\n')
        f.close()

    def load_k_points(self, keyword_savefile):
        # Load k_points from text file
        k_points = []

        # Load eigenvalues
        f = open(
            self.parent_folder + "k_points/" + keyword_savefile + "_" + self.prefix, "r")
        for line in f:
            line_values = line.split()
            k_points.extend(line_values)
        f.close()
        k_points = np.array(k_points, dtype=float).reshape((len(k_points) // 3, 3))

        return k_points

    def calculate_single_eigenvalue(self, i, k_points, hopping_matrix_filename):
        # Calculate eigenvalues, eigvecs, edge particiation, and spin exp values for a single k-point (i)

        # Select k_point i
        k = k_points[i]

        # Calculate eigvals and eigvecs
        current_eigval, current_eigvec = linalg.eigh(self.get_hamiltonian(k, hopping_matrix_filename))

        return current_eigval

    def calculate_eigenvalues(self, k_points, keyword_savefile):
        # Diagonalize the Hamiltonian defined as a hopping matrix, store eigenvalues and eigenvectors

        hopping_matrix_filename = "hopping_matrix/hopping_matrix_" + self.prefix

        try:
            output_file = open(self.parent_folder + hopping_matrix_filename, "r")
            output_file.close()
            print("Existing hopping matrix found, will be used\n")
        except:
            print("No existing hopping matrix found, please define hopping matrix before calculating band structure")
            return None

        ### Diagonalize the Hamiltonian along the selected k_points path ###

        # Time it
        secs_tot = time.time()

        # Parallel execution of calculating eigenvalues (and more) for all kpoints, n_jobs=8 around 3-4 times faster than single core execution
        results = Parallel(n_jobs=8)(delayed(self.calculate_single_eigenvalue)(i, k_points, hopping_matrix_filename) for i in range(len(k_points)))

        # Initialize empty lists for results
        eigvals = []

        # Store results in the lists
        for i in range(len(k_points)):
            eigvals.append(results[i])

        eigvals = np.array(eigvals)
        print("generated and solved all matrices in " + str((time.time() - secs_tot) / 60) + " minutes")

        # Fill up bands to get HOMO, LUMO and Ef
        n_filling = len(k_points)
        HOMO = np.sort(np.reshape(eigvals, n_filling * self.n_bnd))[
            n_filling * self.n_atoms * self.electrons_per_atom - 1]
        LUMO = np.sort(np.reshape(eigvals, n_filling * self.n_bnd))[n_filling * self.n_atoms * self.electrons_per_atom]
        Ef = (HOMO + LUMO) / 2

        print("\nFermi level: " + str(round(Ef, 3)) + "eV")

        # Save eigenvalues and edge participation
        self.save_k_points(k_points, keyword_savefile)
        self.save_eigvals(eigvals, Ef, keyword_savefile)
        print("\nSaved k-points, eigenvalues and edge participation to file")

    def calculate_band_structure(self, symmetry_points_list, resolution, keyword_savefile="band_structure"):
        # Calculate band structure along the defined high symmetry points

        # Make a list of kpoints between high symmetry points
        k_points = np.empty((0, 3))
        for point in range(len(symmetry_points_list) - 1):
            k_points = np.concatenate((k_points, np.linspace(symmetry_points_list[point], symmetry_points_list[point + 1], resolution)))

        # Eigenvalues and eigenvectors calulated and saved to file
        self.calculate_eigenvalues(k_points, keyword_savefile)

    def calculate_constant_energy_surfaces(self, hkl, width_Å, resolution, keyword_savefile="constant_energy_surfaces"):
        # Calculate band structure along the defined high symmetry points, k_space_edges_list = [Gamma, X1, X2],
        # resolution is along Gamma-X1
        # X1 and X2 must be normal, or the plot will be skewed!

        # Make a grid of kpoints in the defined region of k-space
        k_points = np.empty((0, 3))
        surface_normal = np.dot(hkl, self.b_lattice_vectors_ext)
        X1 = np.cross(np.array([0.111, 0.222, 0.333]), surface_normal)
        X1 *= width_Å/np.linalg.norm(X1)
        X2 = np.cross(X1, surface_normal)
        X2 *= width_Å / np.linalg.norm(X2)
        shift = X2 / resolution

        k_points = np.concatenate((k_points, np.linspace(-X1, X1, resolution + 1)))
        for point in range(1, resolution // 2 + 1):
            k_points = np.concatenate(
                (k_points, np.linspace(-X1 + shift * point * 2, X1 + shift * point * 2, resolution + 1)))
            k_points = np.concatenate(
                (k_points, np.linspace(-X1 - shift * point * 2, X1 - shift * point * 2, resolution + 1)))

        # Eigenvalues and eigenvectors calulated and saved to file
        self.calculate_eigenvalues(k_points, keyword_savefile)

    def plot_band_structure(self, symmetry_points_list, keyword_savefile="band_structure", labels=["Gamma", "T", "B", "X", "F", "Gamma", "L"], save_plot=False, E_min=-13, E_max=5, set_Ef=None):
        ### Plot the already calculated band structure along high symmetry path ###

        # Load kpoints and eigenvalues (including edge participation and spin)
        k_points = self.load_k_points(keyword_savefile)
        eigvals, Ef = self.load_eigvals(keyword_savefile)
        resolution = len(k_points) // (len(symmetry_points_list) - 1)

        # Overwrite Ef if wanted
        if set_Ef != None:
            Ef = set_Ef

        # Make a k_path list that holds the distance travelled along the k_path
        k_path = [0]
        for k_nr in range(1, len(k_points)):
            k_path.append(k_path[-1] + np.linalg.norm(k_points[k_nr] - k_points[k_nr - 1]))

        # Plot band structure along entire high symmetry path
        fig = plt.figure(figsize=(12, 10))

        # Loop through bands and plot
        for bnd in range(self.n_bnd):
            plt.plot(k_path, eigvals[:, bnd] - Ef) #, color="tab:blue")

        # Add vertical lines for high symmetry points and horizontal line at Fermi level
        plt.vlines([*k_path[::resolution], k_path[-1]], ymin=np.min(eigvals), ymax=np.max(eigvals), linestyles="--", colors="k")
        plt.hlines(0, xmin=0, xmax=np.max(k_path), linestyles="dotted", colors="b")

        # Add high symmetry points labels
        plt.xticks([*k_path[::resolution], k_path[-1]], labels=labels)

        # Set energy scale limits
        plt.ylim(E_min, E_max)
        plt.tight_layout()

        if save_plot:
            plt.savefig(
                self.parent_folder[:-19] + "plots/" + keyword_savefile + "_" + self.prefix + ".pdf", format="pdf",
                bbox_inches='tight')
        else:
            plt.show()

    def plot_constant_energy_surfaces(self, hkl, width_Å, resolution, keyword_savefile="constant_energy_surfaces",save_plot=False, sigma=0.05, Eb_list=np.array([0.0]), temperature=298, set_Ef=None):
        ### Plot the already calculated constant energy surfaces ###

        """
        Need to implement selection of k_z, either by ARPES Photon energy and internal potential, or by summing all k_z in the 1st BZ
        Change input variables to be X1 and X2 instead perhaps?
        """

        # Load kpoints and eigenvalues (including edge participation and spin)
        k_points = self.load_k_points(keyword_savefile)
        eigvals, Ef = self.load_eigvals(keyword_savefile)

        # Overwrite Ef if wanted
        if set_Ef != None:
            Ef = set_Ef

        # Set high symmetry points based on list input
        surface_normal = np.dot(hkl, self.b_lattice_vectors_ext)
        X1 = np.cross(np.array([0.111, 0.222, 0.333]), surface_normal)
        X1 *= width_Å/np.linalg.norm(X1)
        X2 = np.cross(X1, surface_normal)
        X2 *= width_Å / np.linalg.norm(X2)

        # Energy threshold for including a kpoint in the heatmap_matrix
        boltzmann_energy = 8.62 * 1e-5 * temperature  # Boltzmann constant * temperature

        plot_shape = Eb_list.shape
        fig, axis = plt.subplots(plot_shape[0], plot_shape[1], figsize=np.array(plot_shape) * 6)
        # Loop through all calculated kpoints for all Eb and all of their eigenvalues
        count = 0
        for Eb in Eb_list.reshape(Eb_list.size):

            # List with all kpoints that have an eigenvalue equal Ef - Eb
            heatmap_matrix = []

            for kpoint_nr in range(len(k_points)):
                kpoint = k_points[kpoint_nr]

                kx = np.dot(X1, kpoint)
                ky = np.dot(X2, kpoint)

                fermi_dist = 1 / (np.exp((eigvals[kpoint_nr, :] - Ef) / boltzmann_energy) + 1)
                weight = fermi_dist * np.exp((-(eigvals[kpoint_nr, :] - (Ef - Eb)) ** 2) / (2 * sigma ** 2))
                weight = np.sum(weight)

                heatmap_matrix.append([kx, ky, weight])

            heatmap_matrix = np.array(heatmap_matrix)
            heatmap, xedges, yedges = np.histogram2d(heatmap_matrix[:, 1], heatmap_matrix[:, 0], bins=(resolution + 1, resolution + 1), weights=heatmap_matrix[:, 2])
            extent = [-np.linalg.norm(X1) / 2, np.linalg.norm(X1) / 2, -np.linalg.norm(X2) / 2, np.linalg.norm(X2) / 2]

            ax = axis.reshape(axis.size)[count]
            count += 1

            ax.set_title("E$_{b}$ = " + str(round(Eb, 1)), fontsize=11)
            ax.set_xlabel('Å$^{-1}$', fontsize=11)
            ax.set_ylabel('Å$^{-1}$', fontsize=11)
            ax.tick_params(axis='both', which='major', labelsize=11, direction='out', length=4, width=1.5)

            ax.locator_params(nbins=6, axis='x')
            ax.locator_params(nbins=6, axis='y')

            ax_r = ax.secondary_yaxis('right')
            ax_t = ax.secondary_xaxis('top')
            ax_r.tick_params(axis='y', direction='in')
            ax_t.tick_params(axis='x', direction='in')
            ax_t.locator_params(nbins=6, axis='x')
            ax_r.locator_params(nbins=6, axis='y')
            ax_t.tick_params(axis='both', which='major', labelsize=0, direction='out', length=4, width=1.5)
            ax_r.tick_params(axis='both', which='major', labelsize=0, direction='out', length=4, width=1.5)

            ax.imshow(heatmap, cmap='gray_r', extent=extent)
            make_axes_locatable(ax)

        fig.tight_layout()

        if save_plot:
            plt.savefig(
                self.parent_folder[:-19] + "plots/" + keyword_savefile + "_" + self.prefix + ".pdf", format="pdf",
                bbox_inches='tight')
        else:
            plt.show()

































