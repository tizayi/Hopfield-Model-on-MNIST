import numpy as np
import matplotlib.pyplot as plt
import random as rd
import Hopfield_model as HM
import Finding_patterns as FP
from tensorflow.keras.datasets import mnist

# Getting the MNIST dataset Patterns from tensorflow datasets
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
# Finding the index of the best train patterns for each
idx = FP.find_patterns(X_train,Y_train,20,rdseed=9).astype(int)

# Getting test patterns
idxmat = FP.get_test_patterns(100,Y_train).astype(int)

# List train of patterns
train_Patterns = HM.getspin(X_train[idx,:,:])
Patterns = train_Patterns[[3,6]] 

# Random Initialisation 
#Spins = np.random.randint(2, size=(28,28))
#Spins[Spins==0] = -1

# Noisy pattern initialisation
#Spins = HM.add_noise(Patterns[1],0.1)

# Test pattern intilisation
test_Patterns = HM.getspin(X_train[idxmat,:,:])
Spins=test_Patterns[6,87]

Snew = HM.Model(Spins,Patterns,T=0.2,sweeps=50)
OvLp = HM.overlap(Snew,Patterns[1])
print(OvLp)
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