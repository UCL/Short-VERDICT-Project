% MATLAB script to implement a grid search over parameters delta, Delta and
% b to find scans which optimise Fisher information about each model
% parameter.

% == STEPS

% 1. Define model type, number of paramaters, and parameter index
% 2. Define grid of scan parameters (ensuring gradient constraints are satisfied)
% 3. Evaluate Fisher Information for each set of scan parameters
% 4. Select best scan paramater


% =====================================================================

% 1. Define model type, number of paramaters, and parameter index

% model type
modeltype = 'RDI v1.3';

% Number of parameters
Nparam = 3;

% parameter index 
paramIndx = 1;


% =====================================================================

% 2. Define grid of scan paramaters

% delta
deltaMin = 5;
deltaMax = 35;
deltaN = 7;

deltas = linspace(deltaMin, deltaMax, deltaN);

% Delta
DeltaMin = 10;
DeltaMax = 50;
DeltaN = 9;

Deltas = linspace(DeltaMin, DeltaMax, DeltaN);

% b 
bMin = 500;
bMax = 2500;
bN = 9;

bs = linspace(bMin, bMax, bN);


% Make parameter grids
[varargout{1:3}] = ndgrid(deltas, Deltas, bs);
ScanParamGrid = cat(4, varargout{1:3});

% Flatten grid
ScanParamGridFlat = reshape(ScanParamGrid, deltaN*DeltaN*bN, 3);


% == Ensure gradient constraints are met

% delta < Delta
deltaDeltaBool = (ScanParamGridFlat(:,1) < ScanParamGridFlat(:,2));

ScanParamGridFlat = ScanParamGridFlat(deltaDeltaBool, :);

% G < Gmax
Gmax = 60;

gradstrengthBool = zeros(length(ScanParamGridFlat), 1);

for scanparamIndx = 1:length(ScanParamGridFlat)

    delta = ScanParamGridFlat(scanparamIndx, 1);
    Delta = ScanParamGridFlat(scanparamIndx, 2);
    b = ScanParamGridFlat(scanparamIndx, 3);

    G = stejskal(delta, Delta, bval = b);

    gradstrengthBool(scanparamIndx) = (G < Gmax);

end

gradstrengthBool = logical(gradstrengthBool);

ScanParamGridFlat = ScanParamGridFlat(gradstrengthBool, :);


% =====================================================================

% 3. Evaluate Fisher Information for each set of scan parameters

% Define number of volume fractions for each parameter
nf = 9;

% Define T2 values to iterate over
T2s = [100];

% Define spherical radius
R = 7.5;

% Define TE constant
TEconst = 22;

% Define noise level
sigma = 0.05;

% Initialise array for FIs
FIs = zeros(length(ScanParamGridFlat), 1);

for scanparamIndx = 1:length(ScanParamGridFlat)

    scan_params = ScanParamGridFlat(scanparamIndx, :)

    FI = FisherInformationParameterGrid( ...
        modeltype,...
        paramIndx, ...
        scan_params, ...
        Nparam, ...
        nf = nf, ...
        T2s = T2s, ...
        R = R,...
        TEconst = TEconst, ...
        sigma = sigma);

    FIs(scanparamIndx) = FI;


end