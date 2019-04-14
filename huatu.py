import numpy as np
import matplotlib.pyplot as plt
import pickle
import pandas as pd
from pandas import read_csv
import scipy.io as scio

def read_data(a):
    input=open(a+'.pk1','rb')
    data=pickle.load(input)
    input.close()
    return data


real_data = read_data('data_house18_sum')
real_data = real_data[0:44620,:]
real_data = real_data[36620:,:]#############

print("real_data.shape = ",real_data.shape)

real0 = real_data[:,4:5]#目标电器0
real1 = real_data[:,5:6]#目标电器1
real2 = real_data[:,6:7]#目标电器2
real3 = real_data[:,7:8]#目标电器3
real4 = real_data[:,8:9]#目标电器4

real0 = np.array(real0)
real1 = np.array(real1)
real2 = np.array(real2)
real3 = np.array(real3)
real4 = np.array(real4)

n_train = int(len(real0)*0.8)
#print(len(real0))#44639
#print(int(len(real0)*0.8))#35711

#int(44620*0.8) = 35696
#int(44620*0.2) = 8924

pre_data = pd.read_csv("./res_gru_8000.csv")#(44620,1)
pre_data = pre_data.values
pre_data = np.array(pre_data)

for i in range(len(pre_data[:, 0])):
    for j in range(len(pre_data[0, :])):
        if pre_data[i, j] < 0:
            pre_data[i, j] = 0

#print("***********")
#print(real_data.shape)#(44620,9)
#print(pre_data.shape)#(44620,5)
#print("***********")
#保存真实的数据
dataframe = pd.DataFrame({'1': real_data[:, 4], '2': real_data[:, 5], '3': real_data[:, 6],'4': real_data[:, 7], '5': real_data[:,8]})
columns = ['1', '2', '3', '4', '5']
dataframe.to_csv("real_house18.csv", index=False, columns=columns)
#将模型预测的数据保存在pred_house18_sum.csv
dataframe = pd.DataFrame({'1': pre_data[:, 0], '2': pre_data[:, 1], '3': pre_data[:, 2],'4': pre_data[:, 3], '5': pre_data[:, 4]})
columns = ['1','2', '3', '4', '5']
dataframe.to_csv("pred_house18.csv", index=False, columns=columns)

print(real_data.shape)#(8924, 5)
print(pre_data.shape)#(8924, 5)

print("real_data.shape = ",real_data.shape)
real00 = real_data[:, 4]
real11 = real_data[:, 5]
real22 = real_data[:, 6]
real33 = real_data[:, 7]
real44 = real_data[:, 8]

plt.subplot(2, 2, 1)
plt.plot(pre_data[:, 0], label='1')
plt.title('1')
plt.subplot(2, 2, 2)
plt.plot(pre_data[:, 1], label='2')
plt.title('2')
plt.subplot(2, 2, 3)
plt.plot(real00, label='1-label')
plt.title('1-label')
plt.subplot(2, 2, 4)
plt.plot(real11, label='2-label')
plt.title('2-label')
plt.show()

plt.subplot(2, 3, 1)
plt.plot(pre_data[:,2], label='3')
plt.title('3')

plt.subplot(2, 3, 2)
plt.plot(pre_data[:,3], label='4')
plt.title('4')
plt.subplot(2, 3, 3)
plt.plot(pre_data[:,4], label='5')
plt.title('5')

plt.subplot(2, 3, 4)
plt.plot(real22,label='3-label')
plt.title('3-label')

plt.subplot(2, 3, 5)
plt.plot(real33, label='4-label')
plt.title('4-label')

plt.subplot(2, 3, 6)
plt.plot(real44, label='5-label')
plt.title('5-label')
plt.show()

