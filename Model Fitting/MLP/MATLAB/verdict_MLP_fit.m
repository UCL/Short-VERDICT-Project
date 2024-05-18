% MATLAB Function to apply MLP fitting
function [fIC, fEES, fVASC, R, rmse] = verdict_MLP_fit(schemename, modeltype, Y, opts)

arguments
    schemename % Name of scheme (char array)
    modeltype % Name of model (char array)
    Y % signal data [..., nscheme]

    % == OPTIONS
   
    % Noise model in MLP training
    opts.noisetype = 'Rice'
    opts.sigma0train = 0.05 
    opts.T2train = 10000

    % mask
    opts.mask = []

    % Scheme structure for checking
    opts.scheme = []

    % == Folders
    opts.FolderPath = 'C:\Users\adam\OneDrive - University College London\UCL PhD\PhD Year 1\Projects\VERDICT Screening\Code\VERDICT-Screening\General Code\Model Fitting\MLP\My Tests'
    opts.schemesfolder = "C:\Users\adam\OneDrive - University College London\UCL PhD\PhD Year 1\Projects\VERDICT Screening\Code\VERDICT-Screening\General Code\Model Fitting\MLP\My Tests\Schemes"
    opts.modelsfolder = "C:\Users\adam\OneDrive - University College London\UCL PhD\PhD Year 1\Projects\VERDICT Screening\Code\VERDICT-Screening\General Code\Model Fitting\MLP\My Tests\New MLP Models"
    opts.pythonfolder = "C:\Users\adam\OneDrive - University College London\UCL PhD\PhD Year 1\Projects\VERDICT Screening\Code\VERDICT-Screening\General Code\Model Fitting\MLP\My Tests\Python"

end

% Scheme folder
schemesfolder = char(opts.schemesfolder);%[opts.FolderPath '\Schemes'];



% Load scheme
load([char(schemesfolder) '/' schemename '.mat']);

% Check scheme agrees
if ~isempty(opts.scheme)

    % Check loaded and expected schemes agree 
    if ~isequal([scheme.bval], [opts.scheme.bval]) && isequal([scheme.delta], [opts.scheme.delta])
        error("schemes don't agree!")
    end

end


% models folder
modelsfolder = char(opts.modelsfolder);

% Load some META data about MLP model
load([modelsfolder '/' modeltype '/' schemename '/' opts.noisetype '/T2_' num2str(opts.T2train) '/sigma_' num2str(opts.sigma0train) '/Meta.mat'])

% Radii used in fitting
Rs = FitMeta.TrainingDataMeta.Rs;


% REMOVE NANS AND INFINTIES
Y(isnan(Y)) = 0;
Y(isinf(Y)) = 0;


% Scheme, Data, and Image sizes
nscheme = length(scheme) ;
szY = size(Y) ;
szmap = szY(1:end-1) ;


% Apply mask
if ~isempty(opts.mask)
    % Check mask size
    if szmap == size(opts.mask)    
        Y = Y.*opts.mask;
    else
        disp('Mask size mismatch')
    end
end

% Flatten
Y = reshape(Y,[prod(szmap) nscheme]) ;

%% Run MLP fitting python script

pythonfolder = char(opts.pythonfolder);%[opts.FolderPath '\Python'];

pyfname = [pythonfolder '/callMLP.py' ];

x = pyrunfile( ...
    pyfname, ...
    'x', ...
    signals = Y, ...
    modeltype = modeltype, ...
    schemename = schemename, ...
    noisetype = opts.noisetype,...
    sigma0train = opts.sigma0train, ...
    T2train = opts.T2train,...
    modelsfolder = opts.modelsfolder);
x = double(x);

%% Reformat results

switch modeltype

    case 'Original VERDICT'
    
        fIC = sum(x(:,1:end-2),2);
        fEES = x(:,end-1);
        fVASC = x(:,end);
        R = sum( Rs.*x(:,1:end-2), 2); 

    case 'No VASC VERDICT'
    
        fIC = sum(x(:,1:end-1),2);
        fEES = x(:,end);
        fVASC = zeros(size(fEES));
        R = sum( Rs.*x(:,1:end-1), 2); 

    case 'RDI'
    
        muRs = FitMeta.TrainingDataMeta.muRs;
        fIC = sum(x(:,1:end-1),2);
        fEES = x(:,end);
        fVASC = zeros(size(fEES));
        R = sum( muRs.*x(:,1:end-1), 2); 

    case 'RDI v1.3'
        fIC = x(:,1);
        R = x(:,2);
        fEES = x(:,3);
        fVASC = zeros(size(fEES));

    case 'RDI v1.4'

        fIC = x(:,1);
        R = x(:,2);
        fEES = x(:,4);
        fVASC = zeros(size(fEES));
end


%% Reshape results
if length(szmap)>1
    fIC = reshape(fIC,szmap) ;
    fEES = reshape(fEES,szmap) ;
    fVASC = reshape(fVASC,szmap);
    R = reshape(R, szmap);
end

rmse = zeros(size(fIC));
end