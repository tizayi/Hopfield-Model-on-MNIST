import numpy as np
import matplotlib.pyplot as plt
import Hopfield_model as HM
from tensorflow.keras.datasets import mnist

np.random.seed(2)

# Getting the MNIST dataset Patternsrom tensorflow datasets
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

# Creating matrix of indices 
idxmat=np.zeros((10,10))
for i in range(10):
    idxmat[i,:] = np.random.choice(np.where(Y_train==i)[0],10)

# Defdine a loss function
def get_loss(i,j,idxmat,X_train):
    P1 = HM.getspin(X_train[int(idxmat[i,j]),:,:],thresh=0)
    # Similarity with the same number 
    same_num = 0
    for k in range(len(idxmat[i,:])):
        P2 = HM.getspin(X_train[int(idxmat[i,k]),:,:],thresh=0)
        same_num += np.tensordot(P1,P2)**2
    
    # Similarity with all other numbers
    dif_num = 0
    for m in range(len(idxmat[i,:])):
        for n in range(len(idxmat[:,j])):
            if m != i:
                P2 = HM.getspin(X_train[int(idxmat[m,n]),:,:],thresh=0) 
                dif_num += (1/(len(idxmat[:,j])-1))*np.tensordot(P1,P2)**2
    
    return same_num-dif_num

# Plotting samples
lossspace=np.zeros((10,10))

for i in range(10):
    for j in range(10):
        # Finding loss space
        loss = get_loss(i,j,idxmat,X_train)
        lossspace[i,j]=loss

best_idx = np.zeros(10) 
for y in range(len(idxmat[i,:])):
    best_idx[y] = np.argmin(lossspace[y,:])
print(best_idx)

fig, ax = plt.subplots(10,10,figsize=(8,8))
for i in range(10):
    for j in range(10):
        P = HM.getspin(X_train[int(idxmat[i,j]),:,:],thresh=0)
        ax[i,j].imshow(P)
        ax[i,j].set_xticks([]) 
        ax[i,j].set_yticks([])
        p = int(best_idx[i])
        ax[i,p].title.set_text('best')
plt.show()
