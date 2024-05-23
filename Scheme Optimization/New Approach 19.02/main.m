% MAIN SCRIPT

% Experiment information structure
experiment = struct();

%% Model type
modeltype = 'RDI v1.3';

experiment.modeltype = modeltype;

%% Scheme

% Define scheme
% 
% % ORIGINAL SCHEME
% V01 = [1,2,0,94,2000,1];
% V1 = [23.5, 46.9, 1500, 94, 2000, 3];
% V02 = [1,2,0,94,2000,1];
% V2 = [14, 37.4, 2000, 75, 2000, 3];


% SHORT SCHEME
V01 = [1,2,0, 62, 4000, 1];
V1 = [12, 35, 1000, 62, 4000, 3];
V02 = [1,2,0, 76, 4000, 1];
V2 = [20, 41, 1800, 76, 4000, 3];

V3 = [18, 25, 1400, 60, 4000, 3];

% Build scheme
Vs = [...
    V1;...
    V2;...
    V3...
    ];

% build scheme
scheme = BuildScheme(Vs);

experiment.scheme = scheme;

%% Prior grid

priorfICs = linspace(0.2, 0.8 ,3);
priorRs = linspace(5, 10 ,3);

[varargout{1:2}] = ndgrid(priorfICs, priorRs);
priorgrid = cat(3, varargout{1:2});
priorgrid = reshape(priorgrid, length(priorfICs)*length(priorRs), 2);
nprior = length(priorgrid);

experiment.priorgrid = priorgrid;

%% Tissue information
experiment.T2 = 100;
experiment.sigma0 = 0.05;


%% Options
options = struct();

% Parameter step change used in optimisation
options.Rstepsize = 0.1;
options.fICstepsize = 0.01;


experiment.options = options;


%% Evaluate PDF
experiment = evaluate_pdf(experiment);

%% Evaluate log derivatives
experiment = evaluate_dlogfdR(experiment);
experiment = evaluate_dlogfdfIC(experiment);

%% Evaluate FI matrix
experiment = evaluate_FImatrix(experiment);

%% Evaluate determinant score
% experiment = evaluate_detscore(experiment);

M = experiment.FImatrix;
invM = inv(M)

% score = experiment.detscore
