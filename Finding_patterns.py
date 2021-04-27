import numpy as np
import matplotlib.pyplot as plt
import Hopfield_model as HM
from tensorflow.keras.datasets import mnist



# Getting the MNIST dataset Patternsrom tensorflow datasets
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

# Define a loss function
def get_loss(i,j,idxmat,X_train,N):
    P1 = HM.getspin(X_train[int(idxmat[i,j]),:,:],thresh=0)
    # Similarity with the same number 
    same_num = 0
    for k in range(N):
        P2 = HM.getspin(X_train[int(idxmat[i,k]),:,:],thresh=0)
        same_num += np.tensordot(P1,P2)**2
    
    # Similarity with all other numbers
    dif_num = 0
    for m in range(10):
        for n in range(N):
            if m != i:
                P2 = HM.getspin(X_train[int(idxmat[m,n]),:,:],thresh=0) 
                dif_num += np.tensordot(P1,P2)**2
    return dif_num - N*same_num

def find_patterns(X_train,N,view=False,rd_seed=7):
    
    np.random.seed(7)
    # Creating matrix of indices 
    idxmat = np.zeros((10,N))
    for i in range(10):
        idxmat[i,:] = np.random.choice(np.where(Y_train==i)[0],N)


    # Plotting samples
    lossspace = np.zeros((10,N))

    for i in range(10):
        for j in range(N):
            # Finding loss space
            loss = get_loss(i,j,idxmat,X_train,N)
            lossspace[i,j]=loss

    best_idx = np.zeros(10) 
    best_num = np.zeros(10) 
    for y in range(len(idxmat[:,j])):
        best_idx[y] = np.argmin(lossspace[y,:])
        best_num[y] = idxmat[y,int(best_idx[y])]
    
    if view==True:
        # Plotting best examples
        fig, ax = plt.subplots(10,N,figsize=(8,8))
        for i in range(10):
            for j in range(N):
                P = HM.getspin(X_train[int(idxmat[i,j]),:,:],thresh=0)
                ax[i,j].imshow(P)
                ax[i,j].set_xticks([]) 
                ax[i,j].set_yticks([])
                ax[i,int(best_idx[i])].title.set_text('BEST')
        plt.show()
    return best_num

idx=find_patterns(X_train,20,view=True)