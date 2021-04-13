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

# Getting the overlap
Snew = HM.Training(Spins,Patterns,1)
HM.View(Patterns,Spins,Snew)
plt.show()
print(HM.overlap(Spins,P1))

