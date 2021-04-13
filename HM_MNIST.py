import numpy as np
import matplotlib.pyplot as plt
import random as rd
from Hopfield_model import hopfiled_sweep
from matplotlib.widgets import Button
from tensorflow.keras.datasets import mnist

# Converting to image spins
def getspin(img,thresh=0):
    img = img/255
    img[img > thresh] = 1
    img[img <= thresh] = -1
    return img

# Getting the MNIST dataset Patternsrom tensorflow datasets
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

# Getting a number of patterns
P1 = getspin(X_train[9,:,:])
P2 = getspin(X_train[80,:,:])
P3 = getspin(X_train[23,:,:])
P4 = getspin(X_train[56,:,:])

# List of patterns
P = [P1,P2]

# Random initialisation 
S = np.random.randint(2, size=(28,28))
S[S==0] = -1

# Visualisation 
fig, ax = plt.subplots(2,len(P))
for i,pattern in enumerate(P):
    ax[0, i].imshow(pattern)
ax[1,0].imshow(S)

# Hopfield Sweep updating
for i in range(30):
    Snew,H = hopfiled_sweep(S,2,P)
    S = Snew
    ax[1,1].imshow(Snew)
    plt.draw()
    plt.pause(0.1)
plt.show()

# Testing the confuration 
