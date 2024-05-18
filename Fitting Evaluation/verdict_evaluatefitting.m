% Matlab function to evaluate fIC fitting performance of specified model
% and tissue parameter set

function [bias, variance] = verdict_evaluatefitting(scheme, schemename, modeltype, fittingtype, tissue_params, opts)

arguments

    scheme % Define scan parameters
    schemename % Define scheme name
    modeltype % Name of model used in fitting
    fittingtype % type of fitting to use

    tissue_params.fIC % Tissue parameters for signal simulation
    tissue_params.fEES
    tissue_params.T2
    tissue_params.DEES = 2
    tissue_params.Rs = linspace(0.1,15.1,50)
    tissue_params.fRs = []

    opts.Nrep = 1000

    % Simulation
    opts.simnoisetype = 'Rice'
    opts.NoiseSigma = 0.05 % Options for simulation/fitting
    opts.TEconst = 22

    % Fitting
    opts.NoisyMatrix = false
    opts.noisetype = 'Rice'
    opts.sigma0train = 0.05 % Noise magnitude used in MLP fitting
    opts.T2train = 10000

    opts.solver

end

%% Bits and bobs

nscheme = length(scheme);


%% First, generate signal distributions over scheme

% Volume fractions compartment
tissue_params.fVASC = 1 - tissue_params.fIC - tissue_params.fEES;


[signalDists, signals] = SimulateProstateSignalDistributions( ...
    scheme, ...
    fIC = tissue_params.fIC, ...
    Rs = tissue_params.Rs, ...
    fRs = tissue_params.fRs, ...
    fEES = tissue_params.fEES,...
    DEES = tissue_params.DEES, ...
    fVASC = tissue_params.fVASC, ...
    simnoisetype = opts.simnoisetype,...
    T2 = tissue_params.T2, ...
    sigma = opts.NoiseSigma,...
    TEconst = opts.TEconst);



%% Specify model fitting function


switch char(fittingtype)

    % AMICO FITTING
    case 'AMICO'

        switch char(modeltype)
        
            case 'Original VERDICT'
        
                % specify fitting function and options
                fitfunc = @verdict_fit;
                options.ncompart = 2;
        
            case 'No VASC VERDICT'
        
                % specify fitting function and options
                fitfunc = @verdict_fit;
                options.ncompart = 1;
        
        
            case 'RDI'
        
                % specify fitting function and options
                fitfunc = @RDI_fit;
                options.ncompart = 1;
        
        end

    
    % MLP FITTING
    case 'MLP'

        % specify fitting function and options
        fitfunc = @verdict_MLP_fit;

end

% Further fitting specifications
options.NoisyMatrix = opts.NoisyMatrix;
options.NoiseSigma = opts.NoiseSigma;



%% Iterate over noise instances/signal samples


% Sample signal distributions
signalsamples = zeros(opts.Nrep, nscheme);

for noiseIndx = 1:opts.Nrep

    for ischeme = 1:nscheme
        
        % b=0
        if scheme(ischeme).bval == 0
            signalsamples(noiseIndx, ischeme) = 1;
        % b\=0    
        else
            signalsamples(noiseIndx, ischeme) = sampleDistribution( signalDists(ischeme, :), signals);
        end

    end

end


Y = zeros([1,size(signalsamples)]);
Y(1,:,:) = signalsamples;

%% Apply fitting

switch char(fittingtype)

    case 'AMICO'

        [fIC_fits, fEES_fits, fVASC_fits] = fitfunc( ...
            scheme, ...
            Y, ...
            ncompart = options.ncompart, ...
            NoisyMatrix = options.NoisyMatrix, ...
            NoiseSigma = options.NoiseSigma ...
            );

    case 'MLP'
        
        [fIC_fits, fEES_fits, fVASC_fits] = fitfunc( ...
            schemename, ...
            modeltype,...
            Y, ...
            noisetype = opts.noisetype,...
            sigma0train = opts.sigma0train,...
            T2train = opts.T2train,...
            scheme = scheme...
            );
end
%% Calculate bias and variance

bias = mean(fIC_fits - tissue_params.fIC);
variance = var(fIC_fits);

end


