import numpy as np
import matplotlib.pyplot as plt
import random as rd

def Energy(S,P):
    Np = len(P)
    N = S.size
    Psum = 0
    for pattern in P:
        Psum += np.tensordot(pattern,S)**2
    
    H = -(1/2*N)*(np.sum(Psum)) + Np/2
    return H

# Monte carlo algorithm 
def hopfiled_sweep(S,T,P):
    Ny,Nx = S.shape
    # Calculating The energy of the configuration 
    H = Energy(S,P)
    # Performing 1 sweep 
    for k in range(Nx*Ny//2):
    
        # Choosing a spin to flip
        x = np.random.randint(Nx)
        y = np.random.randint(Ny)
        
        S_flip = S.copy()
        S_flip[y,x] = -S[y,x]
        
        # Calculating the new energy
        H_new = Energy(S_flip,P)
        de = H_new - H
    
        # Deciding on whether to take the flip
        rand_value = rd.random()
        if de <= 0 :
            S = S_flip
            H = H_new
        elif rand_value <= np.exp(-de/T):
            S = S_flip
            H = H_new
    return [S,H]