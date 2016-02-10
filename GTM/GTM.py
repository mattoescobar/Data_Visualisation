import numpy as np
import scipy as sp


class GTM(object):

    def __init__(self, input_data=sp.rand(100, 3), rbf_number=25, rbf_width=1,
                 regularization=1, latent_space_size=100):
        """ Initialization of the GTM procedure

        :param input_data: Data to be visualized, where rows are samples and
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

        :param dimension: Size of each 1D latent coordinates
        :return: z: rectangular lattice
        """
        [x, y] = np.meshgrid(np.linspace(0, 1, dimension),
                             np.linspace(1, 0, dimension))
        x = np.ravel(x)
        x = 2*x - max(x)
        y = np.ravel(y)
        y = 2*y - max(y)
        rectangular_lattice = np.array([x, y])
        return rectangular_lattice

    def gtm_initialization(self):
        """ Generation of GTM components used with a 2D latent space """
        # Create GTM latent space grid vectors
        latent_space_dimension = np.sqrt(self.latent_space_size)
        z = self.gtm_rectangular(latent_space_dimension)
        # Create GTM latent rbf grid vectors
        rbf_dimension = np.sqrt(self.rbf_number)
        mu = self.gtm_rectangular(rbf_dimension)

        return z, mu

test = GTM()
[z2, mu2] = test.gtm_initialization()
