% MATLAB script to create new training data

Ntrain = 10000;

% Noise type
noisetype = 'Rice';

% Sigma0
sigma0 = 0.5;

% T2
T2 = 999;

% Training data folder
savedata = true;
TrainingDataFolder = 'C:\Users\adam\OneDrive - University College London\UCL PhD\PhD Year 1\Projects\Short VERDICT Project\Code\Short-VERDICT-Project\Model Fitting\MLP\Training Data';


%% Schemes

% Define schemes for training
schemenames = {'Original Full'};

schemesfolder = 'C:\Users\adam\OneDrive - University College London\UCL PhD\PhD Year 1\Projects\Short VERDICT Project\Code\Short-VERDICT-Project\Model Fitting\Schemes';
savescheme = true;

%% Models

modeltypes = {'Original VERDICT'};


%% Create VERDICT data

for mindx = 1:length(modeltypes)
    modeltype = modeltypes{mindx};
    for schemeindx = 1:length(schemenames)
        schemename = schemenames{schemeindx};

        createVERDICTdata( ...
            modeltype,...
            schemename,...
            Ntrain = Ntrain,...
            noisetype = noisetype,...
            sigma0 = sigma0,...
            T2 = T2,...
            savescheme = savescheme,...
            schemeParentFolder=schemesfolder,...
            savedata=savedata,...
            dataParentFolder=TrainingDataFolder...
            );

    end
end

