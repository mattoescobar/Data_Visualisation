from GTM import GTM
import matplotlib.pyplot as plt
import scipy.io as sio
from sklearn.preprocessing import scale

# Importing a multi-mode data set from MATLAB
input_structure = sio.loadmat('data_simmm.mat')
input_data = scale(input_structure['Xt'])

# GTM training
test = GTM(input_data)
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

# Similarity matrix calculation and plot for all samples
fig3 = plt.figure()
test.similarity_matrix()

plt.show()

