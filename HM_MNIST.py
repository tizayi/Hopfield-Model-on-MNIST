import numpy as np
import matplotlib.pyplot as plt
import random as rd
import Hopfield_model as HM
from matplotlib.widgets import Button
from tensorflow.keras.datasets import mnist

# Getting the MNIST dataset Patternsrom tensorflow datasets
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

# Getting a number of patterns
P1 = HM.getspin(X_train[9,:,:])
P2 = HM.getspin(X_train[80,:,:])
P3 = HM.getspin(X_train[23,:,:])
P4 = HM.getspin(X_train[56,:,:])

# List of patterns
Patterns = [P1,P4]

# Random initialisation 
Spins = np.random.randint(2, size=(28,28))
Spins[Spins==0] = -1


# Generating overlap vs temp graphs 
temps=np.arange(0.1,2,0.1)
mean_overlaps=np.zeros(len(temps))
for temp in temps:
    overlaps = np.zeros(10)
    for i in range (10):
        Snew = HM.Training(Spins,Patterns,T=temp)
        overlaps[i] = HM.overlap(Snew,P1)
    mean_overlaps[i] = np.mean(overlaps)

# Getting the overlap
plt.plot(temps,mean_overlaps)
plt.show()