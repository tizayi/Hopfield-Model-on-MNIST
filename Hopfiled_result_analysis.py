import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


'''
bar = Bar('Processing', max=100*10*len(Patterns))
counter = np.zeros((10,len(Patterns)))

for test in range(10):
    idxmat = FP.get_test_patterns(100,Y_train).astype(int)
    test_Patterns = HM.getspin(X_train[idxmat,:,:])
    for i,trainp  in enumerate(Patterns):
        for j in range(100):
            Spins = test_Patterns[Patident[i],j]
            Snew = HM.Model(Spins,Patterns,T=0.2,W=W,sweeps=10)
            OvLp = HM.overlap(Snew,trainp)
            if OvLp >= 0.99:
                counter[test,i]+=1
            bar.next()

bar.finish()

df=pd.DataFrame(counter,columns=['3','6','1','0','4'])
df.to_csv("C:/Users/tizay/OneDrive/Documents/GitHub/Hopfield-Model-on-MNIST/HM_results2.csv")
'''


df = pd.read_csv("C:/Users/tizay/OneDrive/Documents/GitHub/Hopfield-Model-on-MNIST/HM_results2.csv")
print(df.mean())
print(df.std())
plt.bar(df.columns[1:].values,df.mean()[1:],yerr=df.std()[1:])
plt.xlabel("Digit")
plt.ylabel("Recall accuracy")

f=np.mean(df.mean()[1:])
s=df.std()[1:]**2
d=(df.mean()[1:]-f)**2
pool_s = np.sqrt(np.sum(s+d/5))

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