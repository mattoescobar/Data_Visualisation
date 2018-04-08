# Grid search investigation of optimal hyperparameters give different metrics.
from GTM_Indexes import GTMIndexes
import numpy as np
from TEP import *

# GTM Hyperparameters
rbf = [9, 16, 25, 36, 49, 64, 81]
rbf_width = [0.25, 0.5, 1., 2., 4.]
regularization = [10**-5, 10**-4, 10**-3, 10**-2, 10**-1, 1., 10.]

for tep_idx in [1, 2, 5, 7, 8, 13]:
    # Import tep data
    [x_training, x_test, len_x0, len_x] = tep_input(tep_idx)

    # Load w and beta
    parameter_data = np.load('gtm_tep' + str(tep_idx) + '.png.npz')
    w_tep = parameter_data['arr_0']
    beta_tep = parameter_data['arr_1']

    # GTM r2 calculation
    r2_distance = []
    r2_neighbor = []
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
                r2_distance.append(training.gtm_distance_index(w_tep[idx], beta_tep[idx], x_training))
                r2_neighbor.append(training.gtm_r2_neighbors(w_tep[idx], beta_tep[idx], x_training))
                np.savez('gtm_r2_tep' + str(tep_idx) + '.npz', r2, r2_distance, r2_neighbor)



