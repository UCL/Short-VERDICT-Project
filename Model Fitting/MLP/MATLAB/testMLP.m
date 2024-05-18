% MATLAB script to load MLP model and test on new simulated

% Define modeltype
modeltype = 'Original VERDICT';

% Define scheme name
schemename = 'Original Full';


% Load scheme
load(['Schemes/' schemename '.mat'])

%% Simulate some data
[params, signals] = createVERDICTdata(modeltype, schemename,  sigma0 = 0.05,  Ntrain = 3, savedata = false, randtype = 'normal');

scatter(linspace(1,10,10), signals)

fIC_sim = sum(params(:,1:end-2), 2);


% Run MLP fitting python
% x = pyrunfile('callMLP.py', 'x', signals = signals, modeltype = modeltype, schemename = schemename);
% x = double(x);
% fIC_MLP = sum(x(:,1:end-2), 2);
% fEES_MLP = x(end-1);
% fVASC_MLP = x(end);
% 
% fIC_MLP-fIC_sim
[fIC_MLP, fEES_MLP, fVASC_MLP] = MLP_fit(schemename, modeltype, signals);


% === Run AMICO fitting

% reshape signals
Y = zeros([1, size(signals)]);
Y(1,:,:) = signals;
[fIC_AMICO, fEES_AMICO, fVASC_AMICO] = verdict_fit(scheme, Y);