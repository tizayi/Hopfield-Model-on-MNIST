import numpy as np
import matplotlib.pyplot as plt
import random as rd
from tensorflow.keras.datasets import mnist

# Getting the MNIST dataset Patternsrom tensorflow datasets
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

# Creating matrix of indices 
idxmat=np.zeros((10,10))
for i in range(10):
    idxmat[i,:] = np.random.choice(np.where(Y_train==i)[0],10)

# Plotting samples
fig, ax = plt.subplots(10,10,figsize=(8,8))
for i in range(10):
    for j in range(10):
        P = X_train[int(idxmat[i,j]),:,:]
        ax[i,j].imshow(P)
plt.show()
