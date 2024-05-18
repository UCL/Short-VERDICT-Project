# Python script to train an MLP on training data generated in MATLAB

import scipy.io
import scipy
import numpy as np
import numpy.matlib
import pickle
import os
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
import skl2onnx



# == LOAD TRAINING DATA

# Specify model type
modeltype = 'Original VERDICT'

# Specify scheme
scheme = 'Original Full'


# Training data path
TrainPath = rf'C:\Users\adam\OneDrive - University College London\UCL PhD\PhD Year 1\Projects\VERDICT Screening\Code\VERDICT-Screening\General Code\Model Fitting\MLP\My Tests\Training Data'

# Load randomised parameter sets
ParamsTrain = scipy.io.loadmat(f'{TrainPath}/{modeltype}/{scheme}/params.mat')['params']
SignalsTrain = scipy.io.loadmat(f'{TrainPath}/{modeltype}/{scheme}/signals.mat')['Signals']


# == SET UP MLP MODEL

mlp = MLPRegressor(hidden_layer_sizes=(150, 150, 150),  activation='relu', solver='adam', alpha=0.001, batch_size=100, learning_rate='adaptive', learning_rate_init=0.001, power_t=0.5, max_iter=1000, shuffle=True, random_state=1, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True, early_stopping=True, validation_fraction=0.20, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
# scaler  = MinMaxScaler(copy=True, feature_range=(0, 1))
# scaler.fit(SignalsTrain)
# TrainParam=scaler.transform(SignalsTrain)


# == TRAIN MLP MODEL
mlp.fit(SignalsTrain, ParamsTrain)



# == SAVE MLP MODEL

FolderPath = rf'C:\Users\adam\OneDrive - University College London\UCL PhD\PhD Year 1\Projects\VERDICT Screening\Code\VERDICT-Screening\General Code\Model Fitting\MLP\My Tests'

# Create directory
path = f'{FolderPath}/MLP Models/{modeltype}/{scheme}'

try:
    os.makedirs(path)
except:
    None

# Save mlp as pickle
pickle.dump(mlp, open(f'{path}/mlp.sav', 'wb'))
# pickle.dump(scaler, open(f'{path}/scaler.sav', 'wb'))


# # Save mlp as onnx
# onx = skl2onnx.to_onnx(mlp, SignalsTrain[0:1])
# with open(f'{path}/mlp.onnx', "wb") as f:
#     f.write(onx.SerializeToString())



# # == TEST

# n=77
# SignalTest = SignalsTrain[n:n+1,:]
# ParamsTest = ParamsTrain[n:n+1,:]

# pred = reg.predict(SignalTest)
# # pred = scaler.inverse_transform(pred)

# fIC_pred = np.sum(pred[0,0:-1])
# fIC_true = np.sum(ParamsTest[0,0:-1])
# print(pred)
# print(ParamsTest)
# print(fIC_pred)
# print(fIC_true)


# n=65
# SignalTest = SignalsTrain[n:n+1,:]
# ParamsTest = ParamsTrain[n:n+1,:]

# pred = reg.predict(SignalTest)
# # pred = scaler.inverse_transform(pred)

# fIC_pred = np.sum(pred[0,0:-1])
# fIC_true = np.sum(ParamsTest[0,0:-1])
# print(pred)
# print(ParamsTest)
# print(fIC_pred)
# print(fIC_true)


# n=33
# SignalTest = SignalsTrain[n:n+1,:]
# ParamsTest = ParamsTrain[n:n+1,:]

# pred = reg.predict(SignalTest)
# # pred = scaler.inverse_transform(pred)

# fIC_pred = np.sum(pred[0,0:-1])
# fIC_true = np.sum(ParamsTest[0,0:-1])
# print(pred)
# print(ParamsTest)
# print(fIC_pred)
# print(fIC_true)