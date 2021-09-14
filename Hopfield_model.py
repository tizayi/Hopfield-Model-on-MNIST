import numpy as np
import matplotlib.pyplot as plt
import random as rd
from tensorflow.keras.datasets import mnist

# More object oriented aproach
class HopfieldModel():
    def __init__(self,storedIdx,thresh=0,temp=0.2):
        # Hyperparameters
        self.temp = temp
        self.thresh = thresh
        
        # Index of all 10 patterns
        (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
        
        # Found using Finding patterns.py
        self.idx = [39155, 7411, 38406, 21607, 28089, 38060, 25559, 55147, 52050, 49358]
        self.allPatterns = self.getspin(X_train[self.idx,:,:])
        self.storedIdx =  storedIdx
        self.storedPatterns = self.allPatterns[storedIdx]
        
        # Initilisation
        self.spins = np.random.randint(2, size=(28,28))
        self.spins[self.spins==0] = -1

    # convert to spins
    def getspin(self,img):
        img = img/255
        img[img > self.thresh] = 1
        img[img <= self.thresh] = -1
        return img

    # Get images for Visualisations
    def getPatterns(self):
        return self.storedPatterns
    
    def getState(self):
        return self.spins
    
    # Changing the Spins
    def addNoise(self,prob):
        num = int(self.spins.size*prob)
        rand = np.arange(28)
        for i in range(num):
            randx = np.random.choice(rand)
            randy = np.random.choice(rand)
            self.spins[randx,randy] = -self.spins[randx,randy]
    
    def randomise(self):
        self.spins=np.random.randint(2, size=(28,28))
        self.spins[self.spins==0] = -1
    
    def setSpins(self,num):
        self.spins = self.allPatterns[num].copy()
    
    # Setting Hyperparameters
    def setTemp(self,temp):
        self.temp = temp

    def setTresh(self,thresh):
        self.thresh = thresh
    
    def changePatterns(self,Patterns):
        self.storedIdx = Patterns
        self.storedPatterns = self.allPatterns[self.storedIdx]

    # Calculate the standard Hopfiled Hamiltonian of a configuration
    def getEnergy(self,spins):
        Np = len(self.storedPatterns)
        N = spins.size
        Psum = 0
        for pattern in self.storedPatterns:
            Psum += np.tensordot(pattern,spins)**2
        H = -(1/(2*N))*(Psum) + Np/2
        return H

    def hopfieldSweep(self,sweeps):
        for i in range(sweeps):
            spins = self.spins.copy()
            Ny,Nx = spins.shape
            # Calculating The energy of the initial configuration 
            H = self.getEnergy(spins)
            # Performing 1 sweep 
            for k in range(Nx*Ny//2):
                # Choosing a spin to flip
                x = np.random.randint(Nx)
                y = np.random.randint(Ny)
                # Flipping said spin
                S_flip = spins.copy()
                S_flip[y,x] = -spins[y,x]
                # Calculating the new energy
                H_new = self.getEnergy(S_flip)
                de = H_new - H
                # Deciding on whether to take the flip
                rand_value = rd.random()
                if de <= 0 :
                    spins = S_flip
                    H = H_new
                elif rand_value <= np.exp(-de/self.temp):
                    spins = S_flip
                    H = H_new
            self.spins = spins
    
    # Getting the overlap
    def overlap(self,number):
        pattern = self.allPatterns[number]
        return abs(1/(self.spins.size)*np.tensordot(self.spins,pattern))

# Pseudo inverse method
class pseudoInverseModel(HopfieldModel):
    def __init__(self,storedIdx,thresh=0,temp=0.2):
        super().__init__(storedIdx,thresh=0,temp=0.2)
        self.weights = self.getWeights()

    def getWeights(self):
        patterns = self.storedPatterns
        # Find Pseudo Inverse
        N = patterns[0].size
        Q = np.zeros((len(patterns),len(patterns)))
        for i in range(len(patterns)):
            for j in range(len(patterns)):
                Q[i,j] = np.tensordot(patterns[i],patterns[j])/N
        Q = np.linalg.inv(Q)
        
        # Find weights
        W = np.zeros((N,N))
        for i in range(len(patterns)):
            for j in range(len(patterns)):
                W += (1/N)*Q[i,j]*np.tensordot(patterns[i].flatten(),patterns[j].flatten() , axes=0)
        return W 

    # Psudo inverse Hamiltonian
    def getEnergy(self,spins):
        patterns = self.storedPatterns
        W = self.weights
        N = patterns[0].size
        H = -0.5*(W*np.tensordot(spins.flatten(), spins.flatten(),\
            axes=0)).sum() + N*np.diag(W).sum()
        return H
    
    # Adjust Weights for new pattern too
    def changePatterns(self,Patterns):
        self.storedIdx = Patterns
        self.storedPatterns = self.allPatterns[self.storedIdx]
        self.weights = self.getWeights()

if __name__=="__main__":
    model = HM.pseudoInverseModel([3,6,5,4])
    model.plotPatterns()
    model.plotState()
    model.hopfieldSweep(10)
    model.plotState()

    print(model.overlap(6))