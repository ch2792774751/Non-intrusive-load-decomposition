import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt
import pickle
from pandas import read_csv
from numpy import concatenate
from eval import *

real = read_csv('real_house18.csv', header=0)
pred = read_csv('pred_house18.csv', header=0)
real=real.values
pred=pred.values

print(np.shape(pred),np.shape(real))
#fenlei=recall_precision_accuracy_f1(pred,real)
mae = mean_absolute_error(pred[:,0], real[:,0])
sae = get_sae(pred[:,0], real[:,0])
#print('fenlei:',fenlei)
print('mae:',mae)
print('sae:',sae)
#print(np.mean(pred) - np.mean(real))
#plt.plot(real[:,0])
#plt.plot(pred[:,0])
#plt.show()
# house18_dae_pred = 'house18_dae_pred.mat'
# scio.savemat(house18_dae_pred, {'house18_dae_pred': dae})

