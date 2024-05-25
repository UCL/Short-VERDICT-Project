import pickle
import numpy as np
# import sklearn

def callMLP(modeltype,
            schemename, 
            signals,
            noisetype = 'Rice',
            sigma0train = 0.05,
            T2train= 10000,
            modelsfolder = rf'C:\Users\adam\OneDrive - University College London\UCL PhD\PhD Year 1\Projects\VERDICT Screening\Code\VERDICT-Screening\General Code\Model Fitting\MLP\My Tests\MLP Models'):
 
    T2train = int(T2train)
    
    # Set directory
    path = f'{modelsfolder}/{modeltype}/{schemename}/{noisetype}/T2_{T2train}/sigma_{sigma0train}'

    print(path)

        
        # Load MLP
    with open(f'{path}/mlp.sav', 'rb') as handle:
        mlp = pickle.load(handle)
        
    # Load scaler
    with open(f'{path}/scaler.sav', 'rb') as handle:
        scaler = pickle.load(handle)   
        
    # Load scaler
    with open(f'{path}/scalerout.sav', 'rb') as handle:
        scalerout = pickle.load(handle)    
         
        
    # Apply scaler
    signals = scaler.transform(signals)
        
    # Reshape if needed    
    if len(np.shape(signals))<2:
        signals = np.expand_dims(signals, axis = 0)
    
    pred = mlp.predict(signals)
    
    pred = scalerout.inverse_transform(pred)

    return pred

x = callMLP(modeltype, schemename, signals, noisetype, sigma0train, T2train, modelsfolder)
