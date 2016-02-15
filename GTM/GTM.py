from __builtin__ import staticmethod

import numpy as np
import scipy as sp


class GTM(object):
    def __init__(self, input_data=sp.rand(100, 3), rbf_number=25, rbf_width=1,
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
        :param z: latent variable space forming the input set
        :param mu: centers of basis functions
        :param sigma: standard deviation of the radii-symmetric Gaussian basis functions
        :return: basis_functions_output: matrix of basis functions output values
        """
        distance = np.dot(np.transpose(z), mu)
        z2 = np.sum(z * z, 0, keepdims=True)
        mu2 = np.sum(mu * mu, 0, keepdims=True)
        distance = np.dot(np.transpose(z2), np.ones((1, mu.shape[1]))) + np.dot(np.ones((z.shape[1], 1)), mu2) - \
            2 * distance
        basis_functions_output = np.exp((-1/(2 * sigma ** 2)) * distance)
        basis_functions_output = np.concatenate((basis_functions_output, np.ones((z.shape[1], 1))), 1)

        return basis_functions_output

    def gtm_initialization(self):
        """ Generation of GTM components used with a 2D latent space """
        # Create GTM latent space grid vectors
        latent_space_dimension = np.sqrt(self.latent_space_size)
        z = self.gtm_rectangular(latent_space_dimension)
        # Create GTM latent rbf grid vectors
        rbf_dimension = np.sqrt(self.rbf_number)
        mu = self.gtm_rectangular(rbf_dimension)
        # Calculate the spread of the basis functions
        sigma = self.rbf_width * (mu[0, 0] - mu[0, 1])
        # Calculate the activations of the hidden unit when fed the latent variable samples
        fi = self.gtm_gaussian_basis_functions(z, mu, sigma)

        return z, mu, fi


test = GTM()
[z3, mu3, fi] = test.gtm_initialization()
