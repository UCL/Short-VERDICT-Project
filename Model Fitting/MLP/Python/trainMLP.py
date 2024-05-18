# Python script to be called from MATLAB to train MLP regressor

# Import libraries
import scipy.io
import numpy as np
import os
import pickle
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
import sys

def trainMLP(
    modeltype,
    schemename,
    noisetype = 'Rice',
    sigma0train = 0.05,
    T2train = 100,
    Nlayer = 3,
    Nnode = 150,
    batchsize = 100,
    TrainPath = rf'C:\Users\adam\OneDrive - University College London\UCL PhD\PhD Year 1\Projects\VERDICT Screening\Code\VERDICT-Screening\General Code\Model Fitting\MLP\My Tests\Training Data',
    FolderPath = rf'C:\Users\adam\OneDrive - University College London\UCL PhD\PhD Year 1\Projects\VERDICT Screening\Code\VERDICT-Screening\General Code\Model Fitting\MLP\My Tests\MLP Models'
):
    
    # # Load randomised parameters and signals for training
    # ParamsTrain = scipy.io.loadmat(f'{TrainPath}/{modeltype}/{schemename}/params_σ={sigma0train}.mat')['params']
    # SignalsTrain = scipy.io.loadmat(f'{TrainPath}/{modeltype}/{schemename}/signals_σ={sigma0train}.mat')['Signals']
    
    T2train = int(T2train)
    ParamsTrain = scipy.io.loadmat(f'{TrainPath}/{modeltype}/{schemename}/{noisetype}/T2_{T2train}/sigma_{sigma0train}/params.mat')['params']
    try:
        InputsTrain = scipy.io.loadmat(f'{TrainPath}/{modeltype}/{schemename}/{noisetype}/T2_{T2train}/sigma_{sigma0train}/inputs.mat')['inputs']   
    except:
        InputsTrain = scipy.io.loadmat(f'{TrainPath}/{modeltype}/{schemename}/{noisetype}/T2_{T2train}/sigma_{sigma0train}/signals.mat')['Signals']   
    
    
    # x=0
    # try:
    #     ParamsTrain = scipy.io.loadmat(f'{TrainPath}/{modeltype}/{schemename}/Noise Model - {noisetype}/T2_{T2train}/sigma_{sigma0train}/params.mat')['params']
    #     try:
    #         InputsTrain = scipy.io.loadmat(f'{TrainPath}/{modeltype}/{schemename}/Noise Model - {noisetype}/T2_{T2train}/sigma_{sigma0train}/inputs.mat')['inputs']   
    #     except:
    #         InputsTrain = scipy.io.loadmat(f'{TrainPath}/{modeltype}/{schemename}/Noise Model - {noisetype}/T2_{T2train}/sigma_{sigma0train}/signals.mat')['Signals']   
    #     x=1
        
    # except:
    #     ParamsTrain = scipy.io.loadmat(f'{TrainPath}/{modeltype}/{schemename}/params_σ={sigma0train}.mat')['params']
    #     try:
    #         InputsTrain = scipy.io.loadmat(f'{TrainPath}/{modeltype}/{schemename}/inputs_σ={sigma0train}.mat')['inputs']   
    #     except:
    #         InputsTrain = scipy.io.loadmat(f'{TrainPath}/{modeltype}/{schemename}/signals_σ={sigma0train}.mat')['Signals']   
            
    hidden_layer_sizes = tuple([Nnode for i in range(Nlayer)])
    
    # Data scaler
    scaler =  MinMaxScaler(copy=True, feature_range=(0, 1))
    scaler.fit(InputsTrain)
    InputsTrain=scaler.transform(InputsTrain)
    
    # Data scaler
    scalerout =  MinMaxScaler(copy=True, feature_range=(0, 1))
    scalerout.fit(ParamsTrain)
    ParamsTrain=scalerout.transform(ParamsTrain)
    
    # Set up MLP regressor
    mlp = MLPRegressor(
        hidden_layer_sizes=hidden_layer_sizes,
        activation='relu',
        solver='adam',
        alpha=0.001,
        batch_size=batchsize,
        learning_rate='adaptive',
        learning_rate_init=0.001,
        power_t=0.5,
        max_iter=1000,
        shuffle=True,
        random_state=1,
        tol=0.0001,
        verbose=False,
        warm_start=False,
        momentum=0.9,
        nesterovs_momentum=True,
        early_stopping=True,
        validation_fraction=0.20,
        beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    

    # Train MLP regressor
    print('Training...')
    mlp.fit(InputsTrain, ParamsTrain)
    
    # == SAVE MLP MODEL
    
    # Create directory
    path = f'{FolderPath}/{modeltype}/{schemename}/{noisetype}/T2_{T2train}/sigma_{sigma0train}'

    try:
        os.makedirs(path)
    except:
        None


    # if x==0:
    #     # Save mlp and scaler as pickle
    #     pickle.dump(mlp, open(f'{path}/mlp_σ={sigma0train}.sav', 'wb'))
    #     pickle.dump(scaler, open(f'{path}/scaler_σ={sigma0train}.sav', 'wb'))
    #     pickle.dump(scalerout, open(f'{path}/scalerout_σ={sigma0train}.sav', 'wb'))
        
    # else:
    # Save mlp and scaler as pickle
    pickle.dump(mlp, open(f'{path}/mlp.sav', 'wb'))
    pickle.dump(scaler, open(f'{path}/scaler.sav', 'wb'))
    pickle.dump(scalerout, open(f'{path}/scalerout.sav', 'wb'))
        
    print(f'Model saved here: {path}')
    

Nlayer = int(Nlayer)
Nnode = int(Nnode)
batchsize = int(batchsize)

# Call training function
trainMLP(
    modeltype,
    schemename,
    noisetype,
    sigma0train,
    T2train,
    Nlayer = Nlayer,
    Nnode = Nnode,
    TrainPath = TrainDataFolder,
    FolderPath = ModelFolder)