import numpy as np
import matplotlib.pyplot as plt
import random as rd
from tensorflow.keras.datasets import mnist

# Getting the MNIST dataset Patternsrom tensorflow datasets
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()


idx = np.random.choice(np.where(Y_train==1)[0],10)
P = X_train[9,:,:]
plt.imshow(P)