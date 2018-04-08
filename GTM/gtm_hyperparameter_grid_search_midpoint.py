# Grid search investigation of optimal hyperparameters given midpoint metrics
from GTM import GTM
from GTM_Indexes import GTMIndexes
import matplotlib.pyplot as plt
import scipy.io as sio
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import time as time

# Loading different data
swiss_data = np.load('Swiss_data.npz')
swiss_input_data = swiss_data['arr_0']
swiss_input_test_data = swiss_data['arr_1']
swiss_data = np.load('swiss_midpoint.npz')
swiss_input_midpoint_data = swiss_data['arr_0']
swiss_input_midpoint_neighbor_data = swiss_data['arr_1']

spherical_data = np.load('Spherical_data.npz')
spherical_input_data = spherical_data['arr_0']
spherical_input_test_data = spherical_data['arr_1']
spherical_data = np.load('spherical_midpoint.npz')
spherical_input_midpoint_data = spherical_data['arr_0']
spherical_input_midpoint_neighbor_data = spherical_data['arr_1']

cube_data = np.load('Cube_data.npz')
cube_input_data = cube_data['arr_0']
cube_input_test_data = cube_data['arr_1']
cube_data = np.load('cube_midpoint.npz')
cube_input_midpoint_data = cube_data['arr_0']
cube_input_midpoint_neighbor_data = cube_data['arr_1']

# GTM Hyperparameters
rbf = [9, 16, 25, 36, 49, 64, 81]
rbf_width = [0.25, 0.5, 1., 2., 4.]
regularization = [10**-5, 10**-4, 10**-3, 10**-2, 10**-1, 1., 10.]

parameter_swiss_data = np.load('gtm_swiss.npz')
w_swiss = parameter_swiss_data['arr_0']
beta_swiss = parameter_swiss_data['arr_1']

# GTM r2 calculation
r2_swiss_midpoint = []
r2_swiss_midpoint_neighbor = []
for i in range(0, len(rbf)):
    for j in range(0, len(rbf_width)):
        for k in range(0, len(regularization)):
            print(i, j, k)

            training = GTMIndexes(swiss_input_data, latent_space_size=900, rbf_number=rbf[i],
                                  regularization=regularization[k], rbf_width=rbf_width[j], iterations=50)
            training.gtm_r2_initialization()
            idx = np.ravel_multi_index((i, j, k), (7, 5, 7))
            r2_swiss_midpoint.append(training.gtm_r2(w_swiss[idx], beta_swiss[idx], swiss_input_midpoint_data))
            r2_swiss_midpoint_neighbor.append(training.gtm_r2(w_swiss[idx], beta_swiss[idx],
                                                              swiss_input_midpoint_neighbor_data))
            np.savez('gtm_r2_swiss_midpoint.npz', r2_swiss_midpoint, r2_swiss_midpoint_neighbor)

parameter_spherical_data = np.load('gtm_spherical.npz')
w_spherical = parameter_spherical_data['arr_0']
beta_spherical = parameter_spherical_data['arr_1']

# GTM r2 calculation
r2_spherical_midpoint = []
r2_spherical_midpoint_neighbor = []
for i in range(0, len(rbf)):
    for j in range(0, len(rbf_width)):
        for k in range(0, len(regularization)):
            print(i, j, k)

            training = GTMIndexes(spherical_input_data, latent_space_size=900, rbf_number=rbf[i],
                                  regularization=regularization[k], rbf_width=rbf_width[j], iterations=50)
            training.gtm_r2_initialization()
            idx = np.ravel_multi_index((i, j, k), (7, 5, 7))
            r2_spherical_midpoint.append(training.gtm_r2(w_spherical[idx], beta_spherical[idx],
                                                         spherical_input_midpoint_data))
            r2_spherical_midpoint_neighbor.append(training.gtm_r2(w_spherical[idx], beta_spherical[idx],
                                                                  spherical_input_midpoint_neighbor_data))
            np.savez('gtm_r2_spherical_midpoint.npz', r2_spherical_midpoint, r2_spherical_midpoint_neighbor)

parameter_cube_data = np.load('gtm_cube.npz')
w_cube = parameter_cube_data['arr_0']
beta_cube = parameter_cube_data['arr_1']

# GTM r2 calculation
r2_cube_midpoint = []
r2_cube_midpoint_neighbor = []
for i in range(0, len(rbf)):
    for j in range(0, len(rbf_width)):
        for k in range(0, len(regularization)):
            print(i, j, k)

            training = GTMIndexes(cube_input_data, latent_space_size=900, rbf_number=rbf[i],
                                  regularization=regularization[k], rbf_width=rbf_width[j], iterations=50)
            training.gtm_r2_initialization()
            idx = np.ravel_multi_index((i, j, k), (7, 5, 7))
            r2_cube_midpoint.append(training.gtm_r2(w_cube[idx], beta_cube[idx], cube_input_midpoint_data))
            r2_cube_midpoint_neighbor.append(training.gtm_r2(w_cube[idx], beta_cube[idx],
                                                             cube_input_midpoint_neighbor_data))
            np.savez('gtm_r2_cube_midpoint.npz', r2_cube_midpoint, r2_cube_midpoint_neighbor)

# # Mean and mode plots for all samples
# fig1 = plt.figure()
# means2 = test.gtm_mean(w_optimal, beta_optimal)
# modes2 = test.gtm_mode(w_optimal)
# plt.scatter(means2[:, 0], means2[:, 1], c=t)
#
# fig2 = plt.figure()
# plt.scatter(modes2[:, 0], modes2[:, 1],  c=t)
#
# # Probability distribution plots for all samples
# fig3 = plt.figure()
# test.gtm_pdf()
#
# # Manifold plot
# manifold_data = np.dot(test.fi,w_optimal)
# X = manifold_data[:, 0].reshape((60, 60))
# Y = manifold_data[:, 1].reshape((60, 60))
# Z = manifold_data[:, 2].reshape((60, 60))
#
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.plot_wireframe(X, Y, Z, rstride=1, cstride=1)
# ax.scatter(test.centered_input_data[:, 0], test.centered_input_data[:, 1], test.centered_input_data[:, 2], c='r')
# plt.show()


