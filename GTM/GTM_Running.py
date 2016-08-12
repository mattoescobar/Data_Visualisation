from GTM import GTM
import matplotlib.pyplot as plt
import scipy.io as sio
from sklearn.preprocessing import scale
import networkx as nx
import numpy as np
from SSIM import compute_ssim
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation

# Importing a multi-mode data set from MATLAB
input_structure = sio.loadmat('swiss_roll_data.mat')
input_data = input_structure['X']
input_data = input_data[0:1000, :]
idx = input_structure['t']
idx = idx[:, 0:1000]
idx = np.argsort(idx)
input_data = input_data[idx[0, :], :]
t = np.arange(input_data.shape[0])

#
# # Plot Figure
# fig1 = plt.figure()
# ax = Axes3D(plt.gcf())
# t = np.arange(input_data.shape[0])
# ax.scatter(input_data[:, 0], input_data[:, 1], input_data[:, 2], c=t)
# plt.show()

# GTM training
test = GTM(input_data, latent_space_size=3600, rbf_number=64, regularization=0.001, rbf_width=2, iterations=100)
[w2, beta2, log_likelihood_evol2] = test.gtm_training()

# Mean and mode plots for all samples
fig1 = plt.figure()
means2 = test.gtm_mean(w2, beta2)
modes2 = test.gtm_mode(w2)
plt.scatter(means2[:, 0], means2[:, 1], c=t)
# plt.savefig("py_mean_225.png")

fig2 = plt.figure()
plt.scatter(modes2[:, 0], modes2[:, 1],  c=t)
# plt.savefig("py_mode_225.png")

# Probability distribution plots for all samples
fig3 = plt.figure()
test.gtm_pdf()
# plt.savefig("py_pdf_225.png")

# # Manifold plot
# X = test.input_reconstructed[:, 0,  0].reshape((60, 60))
# Y = test.input_reconstructed[:, 1,  0].reshape((60, 60))
# Z = test.input_reconstructed[:, 2,  0].reshape((60, 60))
#
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.plot_wireframe(X, Y, Z, rstride=1, cstride=1)
# ax.scatter(test.centered_input_data[:, 0], test.centered_input_data[:, 1], test.centered_input_data[:, 2], c='r')
# plt.show()

# Animated Manifold Learning
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
wframes = []
for i in xrange(0, test.iterations):
    x = test.input_reconstructed[:, 0,  i].reshape((60, 60))
    y = test.input_reconstructed[:, 1,  i].reshape((60, 60))
    z = test.input_reconstructed[:, 2,  i].reshape((60, 60))
    wframe = ax.plot_wireframe(x, y, z, rstride=1, cstride=1)
    wframes.append([wframe])

anim = animation.ArtistAnimation(fig, wframes, interval=20, blit=True)
anim.save('manifold.gif', writer='imagemagick', fps=30, extra_args=['-vcodec', 'libx264'])
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
