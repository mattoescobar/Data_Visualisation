# Grid search investigation of optimal hyperparameters given midpoint metrics
from GTM_Indexes import GTMIndexes
from TEP import *
import numpy as np

# GTM Hyperparameters
rbf = [9, 16, 25, 36, 49, 64, 81]
rbf_width = [0.25, 0.5, 1., 2., 4.]
regularization = [10**-5, 10**-4, 10**-3, 10**-2, 10**-1, 1., 10.]

for tep_idx in [1, 2, 5, 7, 8, 13]:
    # Import tep data
    [x_training, x_test, len_x0, len_x] = tep_input(tep_idx)

    # Load midpoint and nearest neighbor data
    tep_data = np.load('tep_midpoint' + str(tep_idx) + '.npz')
    tep_midpoint_data = tep_data['arr_0']
    tep_midpoint_neighbor_data = tep_data['arr_1']

    # Load w and beta
    parameter_data = np.load('gtm_tep' + str(tep_idx) + '.png.npz')
    w_tep = parameter_data['arr_0']
    beta_tep = parameter_data['arr_1']

    # GTM r2 calculation
    r2_midpoint = []
    r2_midpoint_neighbor = []
    r2 = []
    for i in range(0, len(rbf)):
        for j in range(0, len(rbf_width)):
            for k in range(0, len(regularization)):
                print(i, j, k)

                training = GTMIndexes(x_training, latent_space_size=900, rbf_number=rbf[i],
                                      regularization=regularization[k], rbf_width=rbf_width[j], iterations=50)
                training.gtm_r2_initialization()
                idx = np.ravel_multi_index((i, j, k), (7, 5, 7))
                r2.append(training.gtm_r2(w_tep[idx], beta_tep[idx], x_training))
                r2_midpoint.append(training.gtm_r2(w_tep[idx], beta_tep[idx], tep_midpoint_data))
                r2_midpoint_neighbor.append(training.gtm_r2(w_tep[idx], beta_tep[idx],
                                                            tep_midpoint_neighbor_data))
                np.savez('gtm_r2_tep_midpoint' + str(tep_idx) + '.npz', r2, r2_midpoint, r2_midpoint_neighbor)

