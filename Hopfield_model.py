import numpy as np
import matplotlib.pyplot as plt
import random as rd

# Calculate the energy of a configuration
def Energy(Spins,Patterns):
    Np = len(Patterns)
    N = Spins.size
    Psum = 0
    for pattern in Patterns:
        Psum += np.tensordot(pattern,Spins)**2
    H = -(1/2*N)*(Psum) + Np/2
    return H

# Monte carlo algorithm 
def hopfiled_sweep(Spins,T,Patterns):
    Ny,Nx = Spins.shape
    # Calculating The energy of the initial configuration 
    H = Energy(Spins,Patterns)
    # Performing 1 sweep 
    for k in range(Nx*Ny//2):
    
        # Choosing a spin to flip
        x = np.random.randint(Nx)
        y = np.random.randint(Ny)
        
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
            print('Please At least once')
            Spins = S_flip
            H = H_new
    return [Spins,H]