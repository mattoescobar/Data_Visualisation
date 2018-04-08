# Display optimal hyperparameters given different metrics.
from GTM_Indexes import GTMIndexes
import matplotlib.pyplot as plt
import numpy as np

#
def gtm_results(data, r2_opt, r2_idx, color_range, filename):
    # GTM Hyperparameters
    rbf = [9, 16, 25, 36, 49, 64, 81]
    rbf_width = [0.25, 0.5, 1., 2., 4.]
    regularization = [10**-5, 10**-4, 10**-3, 10**-2, 10**-1, 1., 10.]

    test = GTMIndexes(data, latent_space_size=900, rbf_number=rbf[r2_opt[0]], regularization=regularization[r2_opt[2]],
                      rbf_width=rbf_width[r2_opt[1]], iterations=50)
    [w_optimal, beta_optimal, log_likelihood_evolution] = test.gtm_training()

    #Optimal results
    print('For the swiss data, the optimal values for r2 are:\n')
    print('r2 distance: %f with the following hyperparameters:\n' % r2_idx[np.argmax(r2_idx)])
    print('rbf: %f, rbf width = %f, regularization = %f' % (rbf[r2_opt[0]], rbf_width[r2_opt[1]],
                                                            regularization[r2_opt[2]]))

    # Mean and mode plots for all samples
    fig1 = plt.figure()
    means2 = test.gtm_mean(w_optimal, beta_optimal)
    modes2 = test.gtm_mode(w_optimal)
    plt.scatter(means2[:, 0], means2[:, 1], c=color_range)
    plt.savefig('mean_' + filename + '.png')
    fig2 = plt.figure()
    plt.scatter(modes2[:, 0], modes2[:, 1],  c=color_range)
    plt.savefig('mode_' + filename + '.png')

    # Probability distribution plots for all samples
    fig3 = plt.figure()
    test.gtm_pdf()
    plt.savefig('pdf_' + filename + '.png')

    # Manifold plot
    manifold_data = np.dot(test.fi, w_optimal)
    X = manifold_data[:, 0].reshape((30, 30))
    Y = manifold_data[:, 1].reshape((30, 30))
    Z = manifold_data[:, 2].reshape((30, 30))

    fig4 = plt.figure()
    ax = fig4.add_subplot(111, projection='3d')
    ax.plot_wireframe(X, Y, Z, rstride=1, cstride=1)
    ax.scatter(test.centered_input_data[:, 0], test.centered_input_data[:, 1], test.centered_input_data[:, 2], c='r')
    plt.savefig('manifold_' + filename + '.png')

# Loading swiss roll dataset
swiss_data = np.load('Swiss_data.npz')
swiss_input_data = swiss_data['arr_0']
swiss_input_test_data = swiss_data['arr_1']
swiss_data = np.load('gtm_r2_swiss.npz')
r2_swiss_distance_idx = swiss_data['arr_0']
r2_swiss_neighbors_idx = swiss_data['arr_1']
r2_swiss_idx = swiss_data['arr_2']
r2_swiss_all_idx = r2_swiss_distance_idx*r2_swiss_neighbors_idx*r2_swiss_idx
r2_swiss_distance_opt = np.unravel_index(np.argmax(r2_swiss_distance_idx), (7, 5, 7))
r2_swiss_neighbors_opt = np.unravel_index(np.argmax(r2_swiss_neighbors_idx), (7, 5, 7))
r2_swiss_opt = np.unravel_index(np.argmax(r2_swiss_idx), (7, 5, 7))
r2_swiss_all_opt = np.unravel_index(np.argmax(r2_swiss_all_idx), (7, 5, 7))

t = np.linspace(1, swiss_input_data.shape[0], swiss_input_data.shape[0])

gtm_results(swiss_input_data, r2_swiss_distance_opt, r2_swiss_distance_idx, t, 'swiss_distance')
gtm_results(swiss_input_data, r2_swiss_neighbors_opt, r2_swiss_neighbors_idx, t, 'swiss_neighbors')
gtm_results(swiss_input_data, r2_swiss_opt, r2_swiss_idx, t, 'swiss')
gtm_results(swiss_input_data, r2_swiss_all_opt, r2_swiss_all_idx, t, 'swiss_all')

# Spherical Dataset
spherical_data = np.load('spherical_data.npz')
spherical_input_data = spherical_data['arr_0']
spherical_input_test_data = spherical_data['arr_1']
spherical_data = np.load('gtm_r2_spherical.npz')
spherical_data = np.load('gtm_r2_spherical.npz')
r2_spherical_distance_idx = spherical_data['arr_0']
r2_spherical_neighbors_idx = spherical_data['arr_1']
r2_spherical_idx = spherical_data['arr_2']

r2_spherical_distance_idx = spherical_data['arr_0']
r2_spherical_neighbors_idx = spherical_data['arr_1']
r2_spherical_idx = spherical_data['arr_2']
r2_spherical_all_idx = r2_spherical_distance_idx*r2_spherical_neighbors_idx*r2_spherical_idx

r2_spherical_distance_opt = np.unravel_index(np.argmax(r2_spherical_distance_idx), (7, 5, 7))
r2_spherical_neighbors_opt = np.unravel_index(np.argmax(r2_spherical_neighbors_idx), (7, 5, 7))
r2_spherical_opt = np.unravel_index(np.argmax(r2_spherical_idx), (7, 5, 7))
r2_spherical_all_opt = np.unravel_index(np.argmax(r2_spherical_all_idx), (7, 5, 7))

t = np.append(np.linspace(1, 1, swiss_input_data.shape[0]/2), np.linspace(2, 2, swiss_input_data.shape[0]/2))
gtm_results(spherical_input_data, r2_spherical_distance_opt, r2_spherical_distance_idx, t, 'spherical_distance')
gtm_results(spherical_input_data, r2_spherical_neighbors_opt, r2_spherical_neighbors_idx, t, 'spherical_neighbors')
gtm_results(spherical_input_data, r2_spherical_opt, r2_spherical_idx, t, 'spherical')
gtm_results(spherical_input_data, r2_spherical_all_opt, r2_spherical_all_idx, t, 'spherical_all')

# cube Dataset
cube_data = np.load('cube_data.npz')
cube_input_data = cube_data['arr_0']
cube_input_test_data = cube_data['arr_1']
cube_data = np.load('gtm_r2_cube.npz')
cube_data = np.load('gtm_r2_cube.npz')
r2_cube_distance_idx = cube_data['arr_0']
r2_cube_neighbors_idx = cube_data['arr_1']
r2_cube_idx = cube_data['arr_2']

r2_cube_distance_idx = cube_data['arr_0']
r2_cube_neighbors_idx = cube_data['arr_1']
r2_cube_idx = cube_data['arr_2']
r2_cube_all_idx = r2_cube_distance_idx*r2_cube_neighbors_idx*r2_cube_idx

r2_cube_distance_opt = np.unravel_index(np.argmax(r2_cube_distance_idx), (7, 5, 7))
r2_cube_neighbors_opt = np.unravel_index(np.argmax(r2_cube_neighbors_idx), (7, 5, 7))
r2_cube_opt = np.unravel_index(np.argmax(r2_cube_idx), (7, 5, 7))
r2_cube_all_opt = np.unravel_index(np.argmax(r2_cube_all_idx), (7, 5, 7))

t = []
for i in range(0, 8):
    t = np.append(t, np.linspace(i, i, cube_input_data.shape[0]/8))
gtm_results(cube_input_data, r2_cube_distance_opt, r2_cube_distance_idx, t, 'cube_distance')
gtm_results(cube_input_data, r2_cube_neighbors_opt, r2_cube_neighbors_idx, t, 'cube_neighbors')
gtm_results(cube_input_data, r2_cube_opt, r2_cube_idx, t, 'cube')
gtm_results(cube_input_data, r2_cube_all_opt, r2_cube_all_idx, t, 'cube_all')

plt.show()
