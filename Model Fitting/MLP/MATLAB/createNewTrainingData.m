% MATLAB script to create new training data

Ntrain = 20000;

% Noise type
noisetype = 'Rice';

% Sigma0
sigma0 = 0.05;

% T2
T2 = 10000;

% Training data folder
savedata = true;
TrainingDataFolder = 'C:\Users\adam\OneDrive - University College London\UCL PhD\PhD Year 1\Projects\VERDICT Screening\Code\VERDICT-Screening\General Code\Model Fitting\MLP\My Tests\New Training Data';

% Model folder
ModelFolder = 'C:\Users\adam\OneDrive - University College London\UCL PhD\PhD Year 1\Projects\VERDICT Screening\Code\VERDICT-Screening\General Code\Model Fitting\MLP\My Tests\New MLP Models';

%% Schemes

schemenames = {'Original Full', 'Original ex905003000', 'Short Scheme v1'};
schemesfolder = 'C:\Users\adam\OneDrive - University College London\UCL PhD\PhD Year 1\Projects\VERDICT Screening\Code\VERDICT-Screening\General Code\Model Fitting\MLP\My Tests\New Schemes';
savescheme = true;

%% Models

modeltypes = {'Original VERDICT', 'No VASC VERDICT', 'RDI v1.3'};


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

