# Dimensionality reduction comparison between GTM, SOM, and AE for Tennessee Eastman Process (TEP)

# Importing the libraries
from GTM_Indexes import GTMIndexes
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import pandas as pd

# Importing the training set
dataset = sio.loadmat('TEP_1.mat')
dataset_train = np.concatenate([dataset['X0'], dataset['X']], axis=0)
y_target_train = np.zeros(dataset_train.shape[0])
y_target_train[480:] = 1


# -------------- GTM Training ---------------
test = GTMIndexes(dataset_train, latent_space_size=3600, rbf_number=64, regularization=0.001, rbf_width=2,
                  iterations=50)
[w_optimal, beta_optimal, log_likelihood_evolution] = test.gtm_training()

# Visualising GTM
fig1 = plt.figure()
means2 = test.gtm_mean(w_optimal, beta_optimal)
plt.scatter(means2[:, 0], means2[:, 1], c=y_target_train)
plt.show()

# -------------- AE Training ---------------
from keras.layers import Input, Dense
from keras.models import Model
from sklearn.preprocessing import StandardScaler

# Preparing the datasets
sc = StandardScaler()
X_train = sc.fit_transform(dataset_train)
# Encoding to a 2D map representation
encoding_dim = 2
# Input placeholder
input_data = Input(shape=(dataset_train.shape[1],))
# "encoded" is the encoded representation of the input
encoded = Dense(encoding_dim, activation='relu')(input_data)
# "decoded" is the loss reconstruction of the input
decoded = Dense(dataset_train.shape[1], activation='sigmoid')(encoded)
# Mapping inputs to their reconstruction
autoencoder = Model(input_data, decoded)
# Mapping inputs to their encoded representation
encoder = Model(input_data, encoded)

# Training AE
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
autoencoder.fit(X_train, X_train, epochs=1000, batch_size=30, shuffle=False, validation_data=(X_train, X_train))

# Visualising AE encoding map
encoded_imgs = encoder.predict(X_train)
plt.scatter(encoded_imgs[:, 0], encoded_imgs[:, 1], c=y_target_train)
plt.show()


