import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

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
