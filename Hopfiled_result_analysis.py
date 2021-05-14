import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("C:/Users/tizay/OneDrive/Documents/GitHub/Hopfield-Model-on-MNIST/HM_results.csv")
print(df.mean())
print(df.std())