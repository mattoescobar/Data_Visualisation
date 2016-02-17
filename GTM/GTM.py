from __builtin__ import staticmethod

import numpy as np
import scipy as sp
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn.preprocessing import normalize


class GTM(object):
    def __init__(self, input_data=sp.rand(100, 50), rbf_number=25, rbf_width=1,
                 regularization=1, latent_space_size=100):
        """ Initialization of the GTM procedure

        :param input_data: data to be visualized, where rows are samples and
        columns are features
        """
        self.input_data = input_data
        self.rbf_number = rbf_number
        self.rbf_width = rbf_width
        self.regularization = regularization
        self.latent_space_size = latent_space_size

    @staticmethod
    def gtm_rectangular(dimension):
        """ Generation of a rectangular lattice for GTM 2D latent space
        :param dimension: size of each 1D latent coordinates
        :return: z: rectangular lattice
        """
        [x, y] = np.meshgrid(np.linspace(0, 1, dimension),
                             np.linspace(1, 0, dimension))
        x = np.ravel(x)
        x = 2 * x - max(x)
        y = np.ravel(y)
        y = 2 * y - max(y)
        rectangular_lattice = np.array([x, y])
        return rectangular_lattice

    @staticmethod
    def gtm_gaussian_basis_functions(z, mu, sigma):
        """ Calculation of the Gaussian basis functions for a given input set
        :param z: latent variable space distribution forming the input set
        :param mu: centers of basis functions
        :param sigma: standard deviation of the radii-symmetric Gaussian basis
        functions
        :return: basis_functions_matrix: matrix of basis functions output
        values
        """
        distance = cdist(np.transpose(z), np.transpose(mu), 'sqeuclidean')
        basis_functions_matrix = np.exp((-1/(2 * sigma ** 2)) * distance)
        basis_functions_matrix = np.concatenate((basis_functions_matrix,
                                                 np.ones((z.shape[1], 1))), 1)
        return basis_functions_matrix

    def gtm_pc_initialization(self, z, basis_functions_matrix):
        """ Calculation of weight matrix using principal components
        :param z: latent variable space distribution forming the input set
        :param basis_functions_matrix: matrix of basis functions output
        values
        :return: W: Initialized weight matrix
        :return: beta: Initial beta value
        """
        # Calculation of principal components and their explained variance
        pca_input_data = scale(self.input_data)
        pca = PCA().fit(pca_input_data)
        # Eigenvectors scaled by their respective eigenvalues
        eigenvector = np.dot(pca.components_[:, 0:z.shape[0]],
                             np.diag(np.sqrt(pca.explained_variance_
                                             [0:z.shape[0]])))
        # Normalized latent distribution and weight matrix initialization
        z_norm = normalize(z)
        lhs = basis_functions_matrix
        rhs = np.dot(np.transpose(z_norm), np.transpose(eigenvector))
        w = np.linalg.lstsq(lhs, rhs)[0]
        w[-1, :] = np.mean(pca_input_data, 0)
        # Beta initialization
        beta_matrix = np.dot(basis_functions_matrix, w)
        inter_distance = cdist(beta_matrix, beta_matrix, 'sqeuclidean')
        np.fill_diagonal(inter_distance, "inf")
        meanNN = np.mean(np.min(inter_distance))
        beta = 2/meanNN
        if z.shape[0] < self.input_data.shape[1]:
            beta = min(beta, 1/pca.explained_variance_[z.shape[0]+1])
        return w, beta

    def gtm_initialization(self):
        """ Generation of GTM components used with a 2D latent space """
        # Create GTM latent space grid vectors
        latent_space_dimension = np.sqrt(self.latent_space_size)
        z = self.gtm_rectangular(latent_space_dimension)
        # Create GTM latent rbf grid vectors
        rbf_dimension = np.sqrt(self.rbf_number)
        mu = self.gtm_rectangular(rbf_dimension)
        mu = mu * rbf_dimension/(rbf_dimension - 1)
        # Calculate the spread of the basis functions
        sigma = self.rbf_width * (mu[0, 1] - mu[0, 0])
        # Calculate the activations of the hidden unit when fed the latent
        # variable samples
        fi = self.gtm_gaussian_basis_functions(z, mu, sigma)
        # Generate an initial set of weights
        # [W, beta] = gtm_pci()
        w, beta = self.gtm_pc_initialization(z, fi)
        print np.var(w, 0)
        print np.max(w[0:-1, :], 0)
        return z, mu, fi, w, beta

test = GTM()
# test.input_data = np.transpose(np.array([np.linspace(1, 100, 100),
#                                          np.linspace(2, 101, 100),
#                                          np.linspace(3, 102, 100)]))
[z3, mu3, fi2, w2, beta2] = test.gtm_initialization()
