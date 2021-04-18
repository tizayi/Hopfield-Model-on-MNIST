import numpy as np
import matplotlib.pyplot as plt
import random as rd
import Hopfield_model as HM
from tensorflow.keras.datasets import mnist

# Getting the MNIST dataset Patternsrom tensorflow datasets
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

# Getting a number of patterns
P1 = HM.getspin(X_train[9,:,:])
P2 = HM.getspin(X_train[80,:,:])
P3 = HM.getspin(X_train[23,:,:])
P4 = HM.getspin(X_train[56,:,:])

# List of patterns
Patterns = [P2,P3]

# Random initialisation 
# Spins = np.random.randint(2, size=(28,28))
#Spins[Spins==0] = -1

Spins = HM.add_noise(P3,0.2)

Snew = HM.Training(Spins,Patterns,T=0.2,sweeps=50)
HM.View(Patterns,Spins,Snew)
plt.show()

'''
# Generating overlap vs temp graphs 
temps=np.arange(0.1,2,0.1)
mean_overlaps=np.zeros(len(temps))
for j,temp in enumerate(temps):
    overlaps = np.zeros(20)
    for i in range (20):
        Snew = HM.Training(Spins,Patterns,T=temp)
        overlaps[i] = HM.overlap(Snew,P1)
    mean_overlaps[j] = np.mean(overlaps)

plt.plot(temps,mean_overlaps)
plt.show()
'''