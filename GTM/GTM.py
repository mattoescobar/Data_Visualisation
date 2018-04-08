# Full GTM Implementation
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn.preprocessing import normalize
import scipy.io as sio


class GTM(object):
    def __init__(self, input_data=sp.rand(100, 3), rbf_number=25,
                 rbf_width=1, regularization=1, latent_space_size=3600,
                 iterations=100):
        """ Initialization of the GTM procedure
        :param input_data: data to be visualized, where rows are samples and columns are features
        :param rbf_number: number of radial basis functions
        :param rbf_width: width of radial basis functions
        :param regularization: regularization parameter for maximum likelihood optimization
        :param latent_space_size: size of the grid, considering the total number of grid points
        :param iterations: number of cycle used for maximum likelihood training
        """
        self.input_data = input_data
        self.rbf_number = rbf_number
        self.rbf_width = rbf_width
        self.regularization = regularization
        self.latent_space_size = latent_space_size
        self.iterations = iterations
        self.z = None
        self.fi = None
        self.gtm_distance = np.zeros((self.latent_space_size, self.input_data.shape[0]))
        self.gtm_responsibility = np.zeros((self.latent_space_size, self.input_data.shape[0]))
        self.centered_input_data = scale(self.input_data)
        self.input_reconstructed = np.zeros((self.latent_space_size, self.input_data.shape[1], iterations))

    @staticmethod
    def gtm_rectangular(dimension):
        """ Generation of a rectangular lattice for GTM 2D latent space
        :param dimension: size of each 1D latent coordinates
        :return: rectangular lattice: Latent 2D lattice
        """
        [x, y] = np.meshgrid(np.linspace(0, 1, dimension),  np.linspace(1, 0, dimension))
        x = np.ravel(x)
        x = 2 * x - max(x)
        y = np.ravel(y)
        y = 2 * y - max(y)
        rectangular_lattice = np.array([x, y])
        return rectangular_lattice

    def gtm_gaussian_basis_functions(self, mu, sigma):
        """ Calculation of the Gaussian basis functions for a given input set
        :param mu: centers of basis functions
        :param sigma: standard deviation of the radii-symmetric Gaussian basis functions
        :return: basis_functions_matrix: matrix of basis functions output values
        """
        distance = cdist(np.transpose(self.z), np.transpose(mu), 'sqeuclidean')
        basis_functions_matrix = np.exp((-1 / (2 * sigma ** 2)) * distance)
        basis_functions_matrix = np.concatenate((basis_functions_matrix, np.ones((self.z.shape[1], 1))), 1)
        return basis_functions_matrix

    def gtm_pc_initialization(self):
        """ Calculation of weight matrix using principal components
        :return: w: Initialized weight matrix
        :return: beta: Initial scalar value of the inverse variance common to all components of the mixture
        """
        # Calculation of principal components and their explained variance
        pca = PCA()
        pca.fit(self.centered_input_data)
        # Eigenvectors scaled by their respective eigenvalues
        [eigenvalues, eigenvector] = np.linalg.eig(pca.get_covariance())
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvector = eigenvector[:, idx]
        eigenvector_scaled = np.dot(eigenvector[:, 0:self.z.shape[0]], np.diag(np.sqrt(eigenvalues
                                                                                       [0:self.z.shape[0]])))
        # Normalized latent distribution and weight matrix initialization
        z_norm = np.dot(np.diag(1/np.std(self.z, axis=1)), self.z - np.dot(np.diag(np.mean(self.z, axis=1)),
                                                                           np.ones(self.z.shape)))
        # eigenvector_scaled[:, 1] = - eigenvector_scaled[:, 1]
        lhs = self.fi
        rhs = np.dot(np.transpose(z_norm), np.transpose(eigenvector_scaled))
        w = np.linalg.lstsq(lhs, rhs)[0]
        w[-1, :] = np.mean(self.centered_input_data, 0)
        rhs2 = np.linalg.pinv(rhs)
        w2 = np.dot(np.transpose(lhs), np.transpose(np.linalg.pinv(rhs)))

        # Beta initialization
        beta_matrix = np.dot(self.fi, w)
        inter_distance = cdist(beta_matrix, beta_matrix, 'sqeuclidean')
        np.fill_diagonal(inter_distance, np.inf)
        mean_nearest_neighbor = np.mean(np.min(inter_distance))
        beta = 2 / mean_nearest_neighbor
        if self.z.shape[0] < self.input_data.shape[1]:
            beta = min(beta, 1 / pca.explained_variance_[self.z.shape[0]])
        return w, beta

    def gtm_initialization(self):
        """ Generation of GTM components used with a 2D latent space
        :return: w: Initialized weight matrix
        :return: beta: Initial scalar value of the inverse variance common to all components of the mixture
        """
        # Create GTM latent space grid vectors
        latent_space_dimension = np.sqrt(self.latent_space_size)
        self.z = self.gtm_rectangular(latent_space_dimension)
        self.z = self.z[::-1, ::-1]
        # Create GTM latent rbf grid vectors
        rbf_dimension = np.sqrt(self.rbf_number)
        mu = self.gtm_rectangular(rbf_dimension)
        mu = mu * rbf_dimension / (rbf_dimension - 1)
        mu = mu[::-1, ::-1]
        # Calculate the spread of the basis functions
        sigma = self.rbf_width * np.abs(mu[1, 0] - mu[1, 1])
        # Calculate the activations of the hidden unit when fed the latent variable samples
        self.fi = self.gtm_gaussian_basis_functions(mu, sigma)
        # Generate an initial set of weights [W, beta]
        w, beta = self.gtm_pc_initialization()
        return w, beta

    def gtm_responsibilities(self, beta):
        """ Log likelihood calculation and component responsibilities over a Gaussian mixture
        :param beta: scalar value of the inverse variance common to all components of the mixture
        :return: log_likelihood: log likelihood of data under a gaussian mixture
        """

        dist_corr = np.minimum((self.gtm_distance.max(axis=0) + self.gtm_distance.min(axis=0))/2,
                               self.gtm_distance.min(axis=0)+(700*2/beta))
        for i in range(0, self.gtm_distance.shape[1]):
            self.gtm_distance[:, i] = self.gtm_distance[:, i] - dist_corr[i]
        self.gtm_responsibility = np.exp((-beta / 2) * self.gtm_distance)
        responsibility_sum = np.sum(self.gtm_responsibility, 0)
        self.gtm_responsibility = self.gtm_responsibility / np.transpose(responsibility_sum[:, None])
        log_likelihood = np.sum(np.log(responsibility_sum) + dist_corr * (-beta/2)) + self.gtm_distance.shape[1] * \
            ((self.input_data.shape[1] / 2.) * np.log(beta / (2 * np.pi)) - np.log(self.gtm_distance.shape[0]))
        return log_likelihood

    def gtm_training(self):
        """ Training of the map by updating w and beta over distinct cycles
        :return: w: optimal weight matrix
        :return: beta: optimal scalar value of the inverse variance common to all components of the mixture
        :return: log_likelihood_evol: all log likelihood values for each cycle
        """
        [w, beta] = self.gtm_initialization()
        # Calculate Initial Distances
        self.gtm_distance = cdist(np.dot(self.fi, w), self.centered_input_data, 'sqeuclidean')
        # Training loop
        log_likelihood_evol = np.zeros((self.iterations, 1))
        for i in range(0, self.iterations):
            # Update log likelihood and responsibilities
            log_likelihood = self.gtm_responsibilities(beta)
            log_likelihood_evol[i] = log_likelihood
            # Printing diagnostic info
            print("Cycle: %d\t log likelihood: %f\t Beta: %f\n " % (i, float(log_likelihood), beta))
            # Calculate matrix to be inverted
            lbda = self.regularization * np.eye(self.fi.shape[1], self.fi.shape[1])
            intermediate_matrix = np.dot(np.transpose(self.fi), np.diag(np.sum(self.gtm_responsibility, 1)))
            maximization_matrix = np.dot(intermediate_matrix, self.fi) + lbda / beta
            inv_maximization_matrix = np.linalg.pinv(maximization_matrix)
            w = np.dot(inv_maximization_matrix, np.dot(np.transpose(self.fi), np.dot(self.gtm_responsibility,
                                                                                     self.centered_input_data)))
            self.gtm_distance = cdist(np.dot(self.fi, w), self.centered_input_data, 'sqeuclidean')
            input_data_size = self.input_data.shape[0] * self.input_data.shape[1]
            beta = input_data_size / np.sum(self.gtm_distance * self.gtm_responsibility)
            self.input_reconstructed[:, :, i] = np.dot(self.fi, w)
        return w, beta, log_likelihood_evol

    def gtm_mean(self, w, beta):
        """ Find mean probability density values for each sample in the latent space
        :param w: optimal weight matrix
        :param beta: optimal scalar value of the inverse variance common to all components of the mixture
        """
        self.gtm_distance = cdist(np.dot(self.fi, w), self.centered_input_data, 'sqeuclidean')
        self.gtm_responsibilities(beta)
        means = np.dot(np.transpose(self.gtm_responsibility), np.transpose(self.z))
        return means

    def gtm_mode(self, w):
        """ Find mode probability density values for each sample in the latent space
        :param w: optimal weight matrix
        """
        self.gtm_distance = cdist(np.dot(self.fi, w), self.centered_input_data, 'sqeuclidean')
        min_idx = np.argmin(self.gtm_distance, 0)
        modes = np.transpose(self.z)[min_idx, :]
        return modes

    def gtm_pdf(self):
        """  Plot GTM's probability distribution in the form of a heat map for each sample in the latent space """
        lat_dim = np.sqrt(self.latent_space_size)
        plt.pcolor(np.reshape(self.z[0, :], (lat_dim, lat_dim)), np.reshape(self.z[1, :], (lat_dim, lat_dim)),
                   np.reshape(np.sum(self.gtm_responsibility, 1), (lat_dim, lat_dim)), cmap='magma', vmin=0, vmax=1)
        plt.colorbar()

    def similarity_matrix(self):
        """ Calculate the similarity matrix given all samples used for GTM map training
        :return: similarity_matrix: Matrix assessing the similarity between samples used for GTM map training
        """
        print("Calculating similarity matrix...")
        # Find one tenth of the highest and lowest probability distribution values for each sample in the latent space
        sim_size = int(round(self.latent_space_size/10))
        responsibility_indexes = np.zeros((sim_size * 2, self.input_data.shape[0]))
        corr_input = np.zeros((sim_size * 2, self.input_data.shape[0]))
        for i in range(0, self.input_data.shape[0]):
            responsibility_indexes[0:sim_size, i] = np.argpartition(self.gtm_responsibility[:, i],
                                                                    -sim_size)[-sim_size:]
            responsibility_indexes[sim_size:, i] = np.argpartition(self.gtm_responsibility[:, i], sim_size)[0:sim_size]
        responsibility_indexes = responsibility_indexes.astype(int)
        # Create correlation input matrix for similarity assessment
        for i in range(0, self.input_data.shape[0]):
            corr_input[:, i] = self.gtm_responsibility[responsibility_indexes[:, i], i]
        # Calculate correlation between all samples and build similarity matrix
        similarity_matrix = np.corrcoef(np.transpose(corr_input))
        # Plot heat map of the similarity matrix accordingly
        [x, y] = np.meshgrid(np.linspace(1, self.input_data.shape[0], self.input_data.shape[0]),
                             np.linspace(1, self.input_data.shape[0], self.input_data.shape[0]))
        x = np.ravel(x)
        y = np.ravel(y)
        sim_lat = np.array([x, y])
        print("Plotting color mesh image...")
        plt.pcolormesh(np.reshape(sim_lat[0, :], (self.input_data.shape[0], self.input_data.shape[0])),
                   np.reshape(sim_lat[1, :], (self.input_data.shape[0], self.input_data.shape[0])), similarity_matrix,
                   cmap='magma', vmin=0, vmax=1)
        plt.colorbar()
        plt.axis([x.min(), x.max(), y.min(), y.max()])
        plt.gca().invert_yaxis()
        return similarity_matrix
