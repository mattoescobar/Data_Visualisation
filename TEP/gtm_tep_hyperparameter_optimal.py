# Display optimal hyperparameters given different metrics.
from GTM_Indexes import GTMIndexes
from TEP import *
import matplotlib.pyplot as plt
import numpy as np


def gtm_results(data, data_test, r2_opt, color_range, filename):
    # GTM Hyperparameters
    rbf = [9, 16, 25, 36, 49, 64, 81]
    rbf_width = [0.25, 0.5, 1., 2., 4.]
    regularization = [10**-5, 10**-4, 10**-3, 10**-2, 10**-1, 1., 10.]

    gtm_map = GTMIndexes(data, latent_space_size=900, rbf_number=rbf[r2_opt[0]],
                         regularization=regularization[r2_opt[2]], rbf_width=rbf_width[r2_opt[1]], iterations=50)
    print('Map training...')
    [w_optimal, beta_optimal, log_likelihood_evolution] = gtm_map.gtm_training(quiet=0)

    # Performance metrics
    r2 = gtm_map.gtm_r2(w_optimal, beta_optimal, data)
    rmse = gtm_map.gtm_rmse(w_optimal, beta_optimal, data)
    r2_test = gtm_map.gtm_r2(w_optimal, beta_optimal, data_test)
    rmse_test = gtm_map.gtm_rmse(w_optimal, beta_optimal, data_test)

    print('The optimal result for r2 and rmse are:\n')
    print('r2: %f, rmse: %f with the following hyperparameters: \n' % (r2, rmse))
    print('rbf: %f, rbf width = %f, regularization = %f' % (rbf[r2_opt[0]], rbf_width[r2_opt[1]],
                                                            regularization[r2_opt[2]]))
    print('The test data performance for r2 and rmse are: \n')
    print('r2_test: %f, rmse_test: %f \n' % (r2_test, rmse_test))

    # Mean and mode plots for all samples
    fig1 = plt.figure()
    means2 = gtm_map.gtm_mean(w_optimal, beta_optimal)
    modes2 = gtm_map.gtm_mode(w_optimal)
    plt.scatter(means2[:, 0], means2[:, 1], color=color_range)
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.savefig('mean_' + filename + '.png')
    fig2 = plt.figure()
    plt.scatter(modes2[:, 0], modes2[:, 1],  color=color_range)
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.savefig('mode_' + filename + '.png')

    # Probability distribution plots for all samples
    fig3 = plt.figure()
    gtm_map.gtm_pdf()
    plt.savefig('pdf_' + filename + '.png')

    # # Test data heat map
    # sum_normal = np.sum(gtm_map.gtm_responsibility[:, 1:497], axis=1)
    # sum_outlier = np.sum(gtm_map.gtm_responsibility[:, 497:977], axis=1)
    # idx = sum_normal > sum_outlier
    # idx[idx == True] = 0
    # idx[idx == False] = 1
    # print 'yay'
    # # idx =
    # # lat_dim = np.sqrt(self.latent_space_size)
    # # plt.pcolor(np.reshape(self.z[0, :], (lat_dim, lat_dim)), np.reshape(self.z[1, :], (lat_dim, lat_dim)),
    # #            np.reshape(np.sum(self.gtm_responsibility, 1), (lat_dim, lat_dim)), cmap='magma', vmin=0, vmax=1)
    # #     plt.colorbar()

for tep_idx in [1, 2, 5, 7, 8, 13]:
# for tep_idx in [8, 13]:
    # Import tep data
    [x_training, x_test, len_x0, len_x] = tep_input(tep_idx)
    red = np.array([1, 0, 0])
    blue = np.array([0, 0, 1])
    t = np.vstack((np.matlib.repmat(blue, 497, 1), np.matlib.repmat(red, 480, 1)))

    # Load distance and neighbor data
    tep_data = np.load('gtm_r2_tep_midpoint' + str(tep_idx) + '.npz')
    r2 = tep_data['arr_0']
    r2_distance = tep_data['arr_1']
    r2_neighbor = tep_data['arr_2']
    r2_all = r2*r2_distance*r2_neighbor

    r2_distance_opt = np.unravel_index(np.argmax(r2_distance), (7, 5, 7))
    r2_neighbor_opt = np.unravel_index(np.argmax(r2_neighbor), (7, 5, 7))
    r2_all_opt = np.unravel_index(np.argmax(r2_all), (7, 5, 7))

    gtm_results(x_training, x_test, r2_distance_opt, t, 'tep_distance' + str(tep_idx))
    gtm_results(x_training, x_test, r2_neighbor_opt, t, 'tep_neighbor' + str(tep_idx))
    gtm_results(x_training, x_test, r2_all_opt, t, 'tep_all' + str(tep_idx))

plt.show()

