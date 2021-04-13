import numpy as np
import matplotlib.pyplot as plt
import random as rd

# Converting to greyscale image into spins
def getspin(img,thresh=0):
    img = img/255
    img[img > thresh] = 1
    img[img <= thresh] = -1
    return img

# Calculate the energy of a configuration
def Energy(Spins,Patterns):
    Np = len(Patterns)
    N = Spins.size
    Psum = 0
    for pattern in Patterns:
        Psum += np.tensordot(pattern,Spins)**2
    H = -(1/(2*N))*(Psum) + Np/2
    return H

# Monte carlo algorithm 
def hopfiled_sweep(Spins,Patterns,T):
    Ny,Nx = Spins.shape
    # Calculating The energy of the initial configuration 
    H = Energy(Spins,Patterns)
    # Performing 1 sweep 
    for k in range(Nx*Ny//2):
        # Choosing a spin to flip
        x = np.random.randint(Nx)
        y = np.random.randint(Ny)
        
        # Flipping said spin
        S_flip = Spins.copy()
        S_flip[y,x] = -Spins[y,x]
        
        # Calculating the new energy
        H_new = Energy(S_flip,Patterns)
        de = H_new - H
        # Deciding on whether to take the flip
        rand_value = rd.random()
        if de <= 0 :
            Spins = S_flip
            H = H_new
        elif rand_value <= np.exp(-de/T):
            Spins = S_flip
            H = H_new
    return [Spins,H]

# Hopfield Sweep updating
def Training(Spins,Patterns,T,sweeps=30):
    for i in range(sweeps):
        Snew,H = hopfiled_sweep(Spins,Patterns,T)
        Spins = Snew
    return Snew

# Visualisation 
def View(Patterns,Spins,Snew):
    fig, ax = plt.subplots(2,len(Patterns))
    for i,pattern in enumerate(Patterns):
        ax[0, i].imshow(pattern)
    ax[1,0].imshow(Spins)
    ax[1,1].imshow(Snew)

# Getiing the overlap
def overlap(Spins,Pattern):
    return abs(1/(Spins.size)*np.tensordot(Spins,Pattern))