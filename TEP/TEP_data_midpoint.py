from GTM_Indexes import GTMIndexes
import numpy as np
from TEP import *


for tep_idx in [1, 2, 5, 7, 8, 13]:

    np.random.seed(0)
    # Importing tep data
    [x_training, x_test, len_x0, len_x] = tep_input(tep_idx)

    tep = GTMIndexes(input_data=x_training)
    tep_midpoint = tep.gtm_midpoint()
    tep_midpoint_neighbors = tep.gtm_midpoint_neighbors(neighbors=10)
    np.savez('tep_midpoint' + str(tep_idx) + '.npz', tep_midpoint, tep_midpoint_neighbors)
