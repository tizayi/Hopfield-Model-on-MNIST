import numpy as np
import matplotlib.pyplot as plt
import random as rd
import Hopfield_model as hm
from matplotlib.widgets import Button
from tensorflow.keras.datasets import mnist

# Converting to image spins
def getspin(img,thresh=0):
    img = img/255
    img[img > thresh] = 1
    img[img <= thresh] = -1
    return img

# Getting the MNIST dataset from tensorflow datasets
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

# Getting 2 patterns
P1 = getspin(X_train[9,:,:])
P2 = getspin(X_train[80,:,:])
P3 = getspin(X_train[23,:,:])

# Random initialisation 
S = np.random.randint(2, size=(28,28))
S[S==0] = -1

# Visualisation 
fig, ax = plt.subplots(2,3)
ax[0, 0].imshow(P1)
ax[0,1].imshow(P2)
ax[0,2].imshow(P3)
ax[1,0].imshow(S)

# Hopfield Sweep updating

for i in range(30):
    Snew,H = hm.hopfiled_sweep(S,1,P1,P2,P3,3)
    S = Snew
    ax[1,1].imshow(Snew)
    plt.draw()
    plt.pause(0.5)
plt.show()
print(H)
