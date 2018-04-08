# Routine used to create TEP datasets
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from sklearn.preprocessing import scale


def tep_input(i):
    # Load training and test datasets
    x0 = sio.loadmat("Datasets\\data00.mat")['tedata']
    x0t = sio.loadmat("Datasets\\data00_test.mat")['tedata']

    if i < 10:
        data_name = "0%s" % str(i)
    else:
        data_name = str(i)

    x = sio.loadmat("Datasets\\data%s.mat" % data_name)[
        'tedata']
    xt = sio.loadmat("Datasets\\data%s_test.mat" % data_name)[
        'tedata']

    len_x0 = x0.shape[0] - 3
    len_x = x.shape[0]
    x_training = np.concatenate([x0, x])
    x_test = np.concatenate([x0t, xt])

    # Add time delay to the datasets --> Useful when capturing the process' dynamic behaviour
    x_training1 = x_training[2:-1, :]
    x_training2 = x_training[1:-2, :]
    x_training3 = x_training[0:-3, :]
    x_training = np.hstack((x_training1, x_training2, x_training3))

    x_test1 = x_test[2:-1, :]
    x_test2 = x_test[1:-2, :]
    x_test3 = x_test[0:-3, :]
    x_test = np.hstack((x_test1, x_test2, x_test3))

    return x_training, x_test, len_x0, len_x
    # #

    #
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.scatter(x_position, y_position)
    # plt.show()




