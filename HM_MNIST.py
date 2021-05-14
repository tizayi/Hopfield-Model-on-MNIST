import numpy as np
import matplotlib.pyplot as plt
import random as rd
import Hopfield_model as HM
import Finding_patterns as FP
from tensorflow.keras.datasets import mnist
import pandas as pd
from progress.bar import Bar


# Getting the MNIST dataset Patterns from tensorflow datasets
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
# Finding the index of the best train patterns for each
idx = FP.find_patterns(X_train,Y_train,20,rdseed=9).astype(int)

# List train of patterns
train_Patterns = HM.getspin(X_train[idx,:,:])
Patterns = train_Patterns[[3,6,7,4]]
Patident = [3,6,7,4]
W = HM.Weight(Patterns)
# Random Initialisation 
#Spins = np.random.randint(2, size=(28,28))
#Spins[Spins==0] = -1

# Noisy pattern initialisation
#Spins = HM.add_noise(Patterns[1],0.1)


# Getting test patterns
idxmat = FP.get_test_patterns(100,Y_train).astype(int)

# Test pattern intilisation
test_Patterns = HM.getspin(X_train[idxmat,:,:])
bar = Bar('Processing', max=4000)
counter = np.zeros((10,4))

for test in range(10):
    for i,trainp  in enumerate(Patterns):
        for j in range(100):
            Spins = test_Patterns[Patident[i],j]
            Snew = HM.Model(Spins,Patterns,T=0.2,W=W,sweeps=10)
            OvLp = HM.overlap(Snew,trainp)
            if OvLp >= 0.99:
                counter[test,i]+=1
            bar.next()

bar.finish()

df=pd.DataFrame(counter,columns=['3', '6', '7','4'])
df.to_csv("C:/Users/tizay/OneDrive/Documents/GitHub/Hopfield-Model-on-MNIST/HM_results2.csv")


#HM.View(Patterns,Spins,Snew)
#plt.show()


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