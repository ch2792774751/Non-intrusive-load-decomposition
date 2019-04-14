import matplotlib.pyplot as plt
import pickle
import numpy as np
import pandas as pd
from pandas import read_csv
from sklearn.preprocessing import MinMaxScaler
from numpy import concatenate
import pickle
import random
from keras.models import load_model
import matplotlib.pyplot as plt

def read_data(a):
    input=open(a+'.pk1','rb')
    data=pickle.load(input)
    input.close()
    return data

################读取数据#############################
values = read_data('data_house18_sum')
#[1:2]是总功率数据
#[0:1]是差分数据
values1 = values[:,1:2]#取出总功率数据
values2 = values[:,4:9]#1到5个电器功率数据
print('values1.shape = ',values1.shape)#(44639,1)
print('values2.shape = ',values2.shape)#(44639,5)
values = np.concatenate((values1,values2),axis=1)
print('values.shape = ',values.shape)#(44639,6)
values = values[0:44620,:]#取前44620个数据结

################设置常量###########################
time_step=20      #时间步
rnn_unit=10       #hidden layer units
rnn_unit1=20
batch_size=32     #每一批次训练多少个样例
input_size=4      #输入层维度
output_size=5     #输出层维度
lr=0.0006         #学习率
keep_prob=0.5     #节点不被dropout的概率
num_layers=2      #深层循环神经网络中LSTM结构的层数

################数据处理############################
#归一化
print("######################values.shape = ",values.shape)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)
print("scaled.shape = ",scaled.shape)#(44620,6)

n_train = int(len(values[:,0]) * 0.8)
train_x = scaled[:n_train,0:1]#(35711,1)
train_y = scaled[:n_train,1:6]#(35711,5)
test_x = scaled[:,0:1]#(44639,1)
test_y = scaled[:,1:6]#(44639,5)

print(train_x.shape)#(35692,1)
print(train_y.shape)#(35692,5)
print(test_x.shape)#(44620,1)
print(test_y.shape)#(44620,5)

train_X=train_x#[:,np.newaxis]
test_X=test_x#[:,np.newaxis]
train_xx,train_yy = [],[]
test_xx,test_yy = [],[]

for i in range(len(train_X)-time_step+1):
    x=train_X[i:i+time_step,]
    y=train_y[i:i+time_step,]
    train_xx.append(x.tolist())
    train_yy.append(y.tolist())

for i in range(int(len(test_X)/time_step)):
    x=test_X[i*time_step:(i+1)*time_step]
    y=test_y[i*time_step:(i+1)*time_step]
    test_xx.append(x.tolist())
    test_yy.append(y.tolist())

train_xx = np.array(train_xx)
train_yy = np.array(train_yy)
test_xx = np.array(test_xx)
test_yy = np.array(test_yy)
print("train_xx.shape = ",train_xx.shape)#(35677, 20)
print("train_yy.shape = ",train_yy.shape)#(35677, 20, 5)
print("test_xx.shape = ",test_xx.shape)#(2231, 20)
print("test_yy.shape = ",test_yy.shape)#(2231, 20, 5)
#reshape
train_xx = train_xx.reshape(35677,20,1)
train_yy = train_yy.reshape(35677,20,5)
test_xx = test_xx.reshape(2231,20,1)
test_yy = test_yy.reshape(2231,20,5)
print("train_xx.shape = ",train_xx.shape)#(35677,20,1)
print("train_yy.shape = ",train_yy.shape)#(35677,20,5)
print("test_xx.shape = ",test_xx.shape)#(2231,20,1)
print("test_yy.shape = ",test_yy.shape)#(2231,20,5)



#取出对应的5个数据
#训练集的标签　　　训练集的输入train_xx
train_yy0 = train_yy[:,:,0]
train_yy1 = train_yy[:,:,1]
train_yy2 = train_yy[:,:,2]
train_yy3 = train_yy[:,:,3]
train_yy4 = train_yy[:,:,4]
#测试集的标签　　　测试集的输入test_xx
test_yy0 = test_yy[:,:,0]
test_yy1 = test_yy[:,:,1]
test_yy2 = test_yy[:,:,2]
test_yy3 = test_yy[:,:,3]
test_yy4 = test_yy[:,:,4]

#电器0
model0 = load_model("./gru0.h5")
#test_xx = (2231,20,1)
#test_xx是总功率
#pred0是目标电器0的预测功率
pred0 = model0.predict(test_xx)
print("pred0.shape = ",pred0.shape)#(2231,20,1)
#pred0 = pred0.flatten()#44620
pred0 = pred0.reshape(2231,20)
for i in range(pred0.shape[0]):
    for j in range(pred0.shape[1]):
        if pred0[i][j]<0:
            pred0[i][j]=0
pred0 = pred0.reshape(44620,1)
print("test_yy0.shape = ",test_yy0.shape)#(44620,1) 目标电器0的真实功率
print("pred0.sahpe = ",pred0.shape)#(44620,1)目标电器0的预测功率

#电器1
model1 = load_model("./gru1.h5")
#test_xx = (2231,20,1)
#test_xx是总功率
#pred1是目标电器0的预测功率
pred1 = model1.predict(test_xx)
print("pred1.shape = ",pred1.shape)#(2231,20,1)
#pred1 = pred1.flatten()#44620
pred1 = pred1.reshape(2231,20)
for i in range(pred1.shape[0]):
    for j in range(pred1.shape[1]):
        if pred1[i][j]<0:
            pred1[i][j]=0
pred1 = pred1.reshape(44620,1)
print("test_yy1.shape = ",test_yy1.shape)#(44620,1) 目标电器0的真实功率
print("pred1.sahpe = ",pred1.shape)#(44620,1)目标电器0的预测功率

#电器2
model2 = load_model("./gru2.h5")
#test_xx = (2231,20,1)
#test_xx是总功率
#pred2是目标电器2的预测功率
pred2 = model2.predict(test_xx)
print("pred2.shape = ",pred2.shape)#(2231,20,1)
#pred3 = pred3.flatten()#44620
pred2 = pred2.reshape(2231,20)
for i in range(pred2.shape[0]):
    for j in range(pred2.shape[1]):
        if pred2[i][j]<0:
            pred2[i][j]=0
pred2 = pred2.reshape(44620,1)
print("test_yy2.shape = ",test_yy2.shape)#(44620,1) 目标电器0的真实功率
print("pred2.sahpe = ",pred2.shape)#(44620,1)目标电器0的预测功率


#电器3
model3 = load_model("./gru3.h5")
#test_xx = (2231,20,1)
#test_xx是总功率
#pred3是目标电器2的预测功率
pred3 = model3.predict(test_xx)
print("pred1.shape = ",pred3.shape)#(2231,20,1)
#pred3 = pred3.flatten()#44620
pred3 = pred3.reshape(2231,20)
for i in range(pred3.shape[0]):
    for j in range(pred3.shape[1]):
        if pred3[i][j]<0:
            pred3[i][j]=0
pred3 = pred3.reshape(44620,1)
print("test_yy3.shape = ",test_yy3.shape)#(44620,1) 目标电器0的真实功率
print("pred3.sahpe = ",pred3.shape)#(44620,1)目标电器0的预测功率

#电器4
model4 = load_model("./gru4.h5")
#test_xx = (2231,20,1)
#test_xx是总功率
#pred4是目标电器2的预测功率
pred4 = model4.predict(test_xx)
print("pred4.shape = ",pred4.shape)#(2231,20,1)
#pred4 = pred4.flatten()#44620
pred4 = pred4.reshape(2231,20)
for i in range(pred4.shape[0]):
    for j in range(pred4.shape[1]):
        if pred4[i][j]<0:
            pred4[i][j]=0
pred4 = pred4.reshape(44620,1)
print("test_yy2.shape = ",test_yy4.shape)#(44620,1) 目标电器0的真实功率
print("pred4.sahpe = ",pred4.shape)#(44620,1)目标电器0的预测功率


print(test_xx.shape)
print(pred0.shape)
print(pred0.shape)
print(pred0.shape)
print(pred0.shape)

#合并
test_xx = test_xx.reshape(44620, 1)
data = np.concatenate((test_xx,pred0,pred1,pred2,pred3,pred4),axis=1)#
print("data.shape = ",data.shape)#(44620,6)
#反归一化
data = scaler.inverse_transform(data)#(44620, 6)

#保存预测值
data = data[36620:,1:6]
data0 = data[:,0]      
data1 = data[:,1]
data2 = data[:,2]
data3 = data[:,3]
data4 = data[:,4]

dataframe = pd.DataFrame({'0':data0,'1':data1,'2':data2,'3':data3,'4':data4})
columns = ['0','1','2','3','4']
dataframe.to_csv("res_gru_8000.csv", index=False, columns=columns)
print(len(data0))
