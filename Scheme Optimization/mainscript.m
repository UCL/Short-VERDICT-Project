% MAIN SCRIPT

% Experiment information structure
experiment = struct();

%% Model type
modeltype = 'RDI v1.3';

experiment.modeltype = modeltype;

%% Scheme

% Define scheme
% V01 = [1,2,0, 50, 2482];
V1 = [20, 30, 2000, 72, 2482];
% V1 = [24, 44, 1500, 94, 2500];

% V02 = [1,2,0, 65, 2480];
V2 = [10, 40, 1000, 72, 2482];
% V2 = [12, 34, 2000, 67, 2500];

% Build scheme
Vs = [...
    V2;...
    V1...
    ];

% build scheme
scheme = BuildScheme(Vs);

experiment.scheme = scheme;

%% Prior grid

priorfICs = linspace(0.2, 0.8 ,5);
priorRs = linspace(5, 10 ,5);

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
invM = size(priorgrid,1)*inv(M)

% score = experiment.detscore
