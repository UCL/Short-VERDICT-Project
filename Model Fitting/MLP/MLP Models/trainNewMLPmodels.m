% MATLAB script to train new MLP models

% Noise type
noisetype = 'Rice';

% Sigma0
sigma0 = 0.05;

% T2
T2 = 10000;

% Training data folder
TrainingDataFolder = 'C:\Users\adam\OneDrive - University College London\UCL PhD\PhD Year 1\Projects\VERDICT Screening\Code\VERDICT-Screening\General Code\Model Fitting\MLP\My Tests\New Training Data';

% Model folder
ModelFolder = 'C:\Users\adam\OneDrive - University College London\UCL PhD\PhD Year 1\Projects\VERDICT Screening\Code\VERDICT-Screening\General Code\Model Fitting\MLP\My Tests\New MLP Models';

batchsize = 100;

%% Schemes
schemenames = {'Original Full', 'Original ex905003000', 'Short Scheme v1'};
schemesfolder = 'C:\Users\adam\OneDrive - University College London\UCL PhD\PhD Year 1\Projects\VERDICT Screening\Code\VERDICT-Screening\General Code\Model Fitting\MLP\My Tests\New Schemes';

%% Models
modeltypes = {'Original VERDICT', 'No VASC VERDICT', 'RDI v1.3'};



for mindx = 1:length(modeltypes)
    modeltype = modeltypes{mindx};
    for schemeindx = 1:length(schemenames)
        schemename = schemenames{schemeindx};

        trainMLP( ...
            modeltype,...
            schemename,...
            noisetype=noisetype,...
            sigma0train=sigma0,...
            T2train=T2,...
            batchsize = batchsize,...
            ModelFolder=ModelFolder,...
            TrainDataFolder=TrainingDataFolder ...
            )

    end
end
