from keras.layers import Conv1D, Dense, Input, Flatten, Conv2D, Dropout, Activation, BatchNormalization, \
    Bidirectional, LSTM,merge,GRU,LSTM,AtrousConv2D,add
from keras.models import Model, Sequential
from keras.utils import plot_model
import matplotlib.pyplot as plt
import pickle
import numpy as np
from keras.callbacks import ModelCheckpoint
import numpy as np
import pandas as pd
from pandas import read_csv
from sklearn.preprocessing import MinMaxScaler
from numpy import concatenate
import tensorflow as tf
import pickle
import random
from tensorflow.contrib import rnn
from keras.models import load_model
from matplotlib import pyplot
import keras.layers as KL
import keras.models as KM
import keras.backend as K

config = tf.ConfigProto()
sess = tf.Session(config=config)
from keras.callbacks import ReduceLROnPlateau, CSVLogger, EarlyStopping
lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.1e-6)
early_stopper = EarlyStopping(min_delta=0.01, patience=100)

def read_data(a):
    input=open(a+'.pk1','rb')
    data=pickle.load(input)
    input.close()
    return data

################读取数据#############################
values = read_data('data_house18_sum')
#[1:2]是总功率数据
#[0:1]是差分数据
values1 = values[:,1:2]#总功率数据
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
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)
print("scaled.shape = ",scaled.shape)#(44620,6)

n_train = int(len(values[:,0]) * 0.8)
train_x = scaled[:n_train,0:1]#(35711,1)
train_y = scaled[:n_train,1:7]#(35711,5)
test_x = scaled[:,0:1]#(44639,1)
test_y = scaled[:,1:7]#(44639,5)

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

#残差模块
def residual_block(filters,x,stride = 1):

    resiual = x
    #BN函数
    #out = BatchNormalization()(x)
    #激活函数
    #out1 = Activation('relu')(out)
    out = Conv1D(filters = int(filters / 4),kernel_size = 1,strides = 1,padding = 'same')(x)
    out = BatchNormalization()(out)
    out = Activation('relu')(out)


    out = Conv1D(filters = int(filters / 4),kernel_size = 3,strides = 1,padding = 'same')(out)
    out = BatchNormalization()(out)
    out = Activation('relu')(out)

    out = Conv1D(filters = filters,kernel_size = 1,strides = 1,padding = 'same')(out)
    out = BatchNormalization()(out)

    if out.shape[-1] != filters or stride == 1:
        residual = Conv1D(filters = filters,kernel_size = 3,strides = 1,padding = 'same')(x)
    out = add([residual,out])#merge(mode = 'sum')([residual,out])
    return out

#############################
#基于空洞卷积的改进深度残差网络模型
def resnet(sequence_len = 20):
    #输入维度4维与2维
    x = Input(shape = [sequence_len,1])
    #常规卷积Conv2D 空洞卷积AtrousConv2D
    conv1 = Conv2D(filters = 30,kernel_size = [5,1],strides = [1,1],padding = 'same')(x)
    bn = BatchNormalization()(conv1)
    out = Activation('relu')(bn)
    #3个残差块也可以设置多个残差块
    residual_block1 = residual_block(filters = 30,x = out)
    residual_block2 = residual_block(filters = 40,x = residual_block1)
    residual_block3 = residual_block(filters = 50,x = residual_block2)
    #全局平均池化层
    #out = MaxPooling2D(pool_size=(2, 1))(residual_block3)
    out = Flatten()(residual_block3)
    #全连接层
    out = Dense(units = sequence_len)(out)
    model = Model(x,out)
    #模型编译　优化器　损失函数
    model.compile(optimizer = 'adam',loss = 'mse')
    return model

#############################



#基于keras的GRU
def build_convGRU_model(sequence_len,layers):#10,20,20
    x = Input(shape = [sequence_len,1])
    #常规卷积Conv2D 空洞卷积AtrousConv2D
    C1 = Conv1D(filters =  6, kernel_size = 1, strides=1, padding="SAME")(x)#(20,1)
    C2 = Conv1D(filters = 10, kernel_size = 3, strides=1, padding="SAME")(x)#(20,1)
    C3 = Conv1D(filters = 14, kernel_size = 5, strides=1, padding="SAME")(x)#(20,1)

    C4 = KL.Concatenate(axis = -1)([C1,C2,C3])

    #out = BatchNormalization()(out)#(20,1)
    #out = Activation('relu')(out)#(20,1)
    #残差块或者多个残差块
    C = residual_block(filters = 30,x = C4)#(20,1)

    #第一层GRU
    GRU1_out = GRU(input_dim=layers[0], output_dim=layers[1], return_sequences=True)(C)
    #产生5个分支，用于5种电器的分解
    #第二层GRU
    GRU2_out1 = GRU(input_dim=layers[0], output_dim=layers[1], return_sequences=True)(GRU1_out)
    GRU2_out2 = GRU(input_dim=layers[0], output_dim=layers[1], return_sequences=True)(GRU1_out)
    GRU2_out3 = GRU(input_dim=layers[0], output_dim=layers[1], return_sequences=True)(GRU1_out)
    GRU2_out4 = GRU(input_dim=layers[0], output_dim=layers[1], return_sequences=True)(GRU1_out)
    GRU2_out5 = GRU(input_dim=layers[0], output_dim=layers[1], return_sequences=True)(GRU1_out)
    #第三层GRU
    GRU3_out1 = GRU(input_dim=layers[0], output_dim=layers[1], return_sequences=True)(GRU2_out1)
    GRU3_out2 = GRU(input_dim=layers[0], output_dim=layers[1], return_sequences=True)(GRU2_out2)
    GRU3_out3 = GRU(input_dim=layers[0], output_dim=layers[1], return_sequences=True)(GRU2_out3)
    GRU3_out4 = GRU(input_dim=layers[0], output_dim=layers[1], return_sequences=True)(GRU2_out4)
    GRU3_out5 = GRU(input_dim=layers[0], output_dim=layers[1], return_sequences=True)(GRU2_out5)
    #拉伸
    out1 = Flatten()(GRU3_out1)
    out2 = Flatten()(GRU3_out2)
    out3 = Flatten()(GRU3_out3)
    out4 = Flatten()(GRU3_out4)
    out5 = Flatten()(GRU3_out5)
    #5个全连接层
    out1 = Dense(units=sequence_len)(out1)
    out2 = Dense(units=sequence_len)(out2)
    out3 = Dense(units=sequence_len)(out3)
    out4 = Dense(units=sequence_len)(out4)
    out5 = Dense(units=sequence_len)(out5)
    #5个分解模型
    model1 = Model(x, out1)
    model1.compile(optimizer='adam', loss='mse')
    model2 = Model(x, out2)
    model2.compile(optimizer='adam', loss='mse')
    model3 = Model(x, out3)
    model3.compile(optimizer='adam', loss='mse')
    model4 = Model(x, out4)
    model4.compile(optimizer='adam', loss='mse')
    model5 = Model(x, out5)
    model5.compile(optimizer='adam', loss='mse')
    return (model1,model2,model3,model4,model5)


model0,model1, model2, model3, model4 = build_convGRU_model(20,[10,20,20])

#model0.summary()
#model1.summary()
#model2.summary()
#model3.summary()
#model4.summary()

#取出对应的5个电器功率数据
#训练集
train_yy0 = train_yy[:,:,0]
train_yy1 = train_yy[:,:,1]
train_yy2 = train_yy[:,:,2]
train_yy3 = train_yy[:,:,3]
train_yy4 = train_yy[:,:,4]
#测试集
test_yy0 = test_yy[:,:,0]
test_yy1 = test_yy[:,:,1]
test_yy2 = test_yy[:,:,2]
test_yy3 = test_yy[:,:,3]
test_yy4 = test_yy[:,:,4]

#模型0
model0.fit(train_xx,train_yy0,epochs=100, batch_size=32,verbose=1,shuffle=True,callbacks=[lr_reducer, early_stopper])
model0.save("./gru0.h5")
#模型1
model1.fit(train_xx,train_yy1,epochs=100, batch_size=32,verbose=1,shuffle=True,callbacks=[lr_reducer, early_stopper])
model1.save("./gru1.h5")
#模型2
model2.fit(train_xx,train_yy2,epochs=100, batch_size=32,verbose=1,shuffle=True,callbacks=[lr_reducer, early_stopper])
model2.save("./gru2.h5")
#模型3
model3.fit(train_xx,train_yy3,epochs=100, batch_size=32,verbose=1,shuffle=True,callbacks=[lr_reducer, early_stopper])
model3.save("./gru3.h5")
#模型4
model4.fit(train_xx,train_yy4,epochs=100, batch_size=32,verbose=1,shuffle=True,callbacks=[lr_reducer, early_stopper])
model4.save("./gru4.h5")
print("GRU模型训练完成，并且已经保存")

