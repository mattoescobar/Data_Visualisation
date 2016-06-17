from GTM import GTM
import matplotlib.pyplot as plt
import scipy.io as sio
from sklearn.preprocessing import scale
import networkx as nx
import numpy as np
from SSIM import compute_ssim

# Importing a multi-mode data set from MATLAB
input_structure = sio.loadmat('SwissRoll.mat')
input_data = input_structure['X_data']
input_data = np.transpose(input_data)

# GTM training
test = GTM(input_data, latent_space_size=3600, rbf_number=49, regularization=0.1, rbf_width=1, iterations=100)
[w2, beta2, log_likelihood_evol2] = test.gtm_training()

# Mean and mode plots for all samples
fig1 = plt.figure()
means2 = test.gtm_mean(w2, beta2)
modes2 = test.gtm_mode(w2)
plt.plot(means2[:, 0], means2[:, 1], 'b.')
plt.plot(modes2[:, 0], modes2[:, 1], 'ro', mfc='None')

# Probability distribution plots for all samples
fig2 = plt.figure()
test.gtm_pdf()
plt.show()

# # Similarity matrix calculation and plot for all samples
# fig3 = plt.figure()
# simple_matrix = test.similarity_matrix()
# np.savez("similarity_matrix", simple_matrix)
#
# # Similarity matrix calculation using all info from GTM map
# lat_dim = np.sqrt(test.latent_space_size)
# full_matrix = np.zeros((input_data.shape[0], input_data.shape[0]))
# for i in xrange(0, input_data.shape[0]):
#     print i
#     for j in xrange(0, input_data.shape[0]):
#         img1 = test.gtm_responsibility[:, i].reshape((lat_dim, lat_dim))
#         img2 = test.gtm_responsibility[:, j].reshape((lat_dim, lat_dim))
#         full_matrix[i, j] = compute_ssim(img1, img2)
#
# np.savez("similarity_matrix", simple_matrix, full_matrix)
#
# # Create network structure using NetworkX
# graph = nx.from_numpy_matrix(full_matrix)
# np.savez("graph_test", graph)
