% MATLAB function to train an MLP to fit VERDICT model
% Function calls python script to train and export MLP model
function trainMLP(modeltype, schemename, opts)


arguments
    
    modeltype % Define model type
    schemename % Define scheme
    
    % Options

    
    % Noise
    opts.noisetype = 'Ratio'
    opts.sigma0train = 0.05 % 
    opts.T2train = 100 % (ms) 


    % MLP Architecture
    opts.Nlayer = 3
    opts.Nnode = 150
    
    % MLP training
    opts.batchsize = 100

    % Files
    opts.pythonfolder = 'C:\Users\adam\OneDrive - University College London\UCL PhD\PhD Year 1\Projects\Short VERDICT Project\Code\Short-VERDICT-Project\Model Fitting\MLP\Python'
    opts.TrainDataFolder = 'C:\Users\adam\OneDrive - University College London\UCL PhD\PhD Year 1\Projects\Short VERDICT Project\Code\Short-VERDICT-Project\Model Fitting\MLP\Training Data'
    opts.ModelFolder = 'C:\Users\adam\OneDrive - University College London\UCL PhD\PhD Year 1\Projects\Short VERDICT Project\Code\Short-VERDICT-Project\Model Fitting\MLP\MLP Models'
end


% Call python script to train mlp
pyfname = [opts.pythonfolder '/trainMLP.py' ];

pyrunfile(pyfname, ...
    modeltype = modeltype, ...
    schemename = schemename, ...
    noisetype = opts.noisetype,...
    sigma0train = opts.sigma0train,...
    T2train = opts.T2train,...
    Nlayer = opts.Nlayer, ...
    Nnode = opts.Nnode, ...
    batchsize = opts.batchsize, ...
    TrainDataFolder = opts.TrainDataFolder, ...
    ModelFolder = opts.ModelFolder...
    );


%% META data

% Load training data meta data
xxx = 0;
% try
%     load([char(opts.TrainDataFolder) '/' modeltype '/' schemename '/Noise Model - ' opts.noisetype '/T2_' num2str(opts.T2) '/sigma_' num2str(opts.sigma0) '/Meta.mat'])
%     xxx=1;
% catch
%     load([char(opts.TrainDataFolder) '/' modeltype '/' schemename '/Meta_σ=' num2str(opts.sigma0train) '.mat'])
% end
load([char(opts.TrainDataFolder) '/' modeltype '/' schemename '/' opts.noisetype '/T2_' num2str(opts.T2train) '/sigma_' num2str(opts.sigma0train) '/Meta.mat'])
xxx=1;

FitMeta = struct();
FitMeta.train_complete_time = datetime();
FitMeta.TrainingDataMeta = Meta;
FitMeta.Nlayer = opts.Nlayer;
FitMeta.Nnode = opts.Nnode;
FitMeta.batchsize = opts.batchsize;


save([char(opts.ModelFolder) '/' modeltype '/' schemename '/' opts.noisetype '/T2_' num2str(opts.T2train) '/sigma_' num2str(opts.sigma0train) '/Meta.mat'], 'FitMeta');


% switch xxx
%     case 0
%         % Save MLP fitting meta data
%         save([char(opts.ModelFolder) '/' modeltype '/' schemename '/Meta_σ=' num2str(opts.sigma0train) '.mat'], 'FitMeta');
%     case 1
%         % Save MLP fitting meta data
%         save([char(opts.ModelFolder) '/' modeltype '/' schemename '/Meta.mat'], 'FitMeta');
% end



end