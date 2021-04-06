import numpy as np
import matplotlib.pyplot as plt
import random as rd

def Energy(S,P1,P2,P3,Np):
    N = S.size
    H = -(1/2*N)*(np.sum(np.tensordot(P1,S)**2 + np.tensordot(P2,S)**2) + np.tensordot(P3,S)**2) + Np/2
    return H

# Monte carlo algorithm 
def hopfiled_sweep(S,T,P1,P2,P3,Np):
    Ny,Nx = S.shape
    # Calculating The energy of the configuration 
    H = Energy(S,P1,P2,P3,Np)
    # Performing 1 sweep 
    for k in range(Nx*Ny//2):
    
        # Choosing a spin to flip
        x = np.random.randint(Nx)
        y = np.random.randint(Ny)
        
        S_flip = S.copy()
        S_flip[y,x] = -S[y,x]
        
        # Calculating the new energy
        H_new = Energy(S_flip,P1,P2,P3,Np)
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