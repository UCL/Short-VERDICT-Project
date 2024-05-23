function FI = FisherInformationParameterGrid(modeltype, paramIndx, scan_params, Nparam, opts)


arguments
    modeltype % Model type to evaluate
    paramIndx % Model parameter to evaluate Fisher Information 
    scan_params % Vector of scan parameters [delta, Delta, b]
    Nparam % Number of model parameters
    
    % verdict: [fR1, fR2, ..., fEES, fVASC]
    % no VASC: [fR1, fR2, ..., fEES]
    % RDI: [fRdist1, fRdist2, ..., fEES]

    % == Options
    opts.fmin = 0 % minimum volume fraction of each parameter
    opts.fmax = 1 % Maximum volume fraction of each parameter
    opts.nf = 6 % Number of volume fractions for each parameter
    opts.R = 7.5 %
    opts.T2s = [100] % T2 values
    opts.TEconst = 22 % TE = delta + Delta + TEconst 
    opts.sigma = 0.05 % Noise level
    opts.paramstepsize = 0.01 % step size when calculating partial derivatives

end


% == STEPS

% 1. Configure tissue paramater grid (with sum(fi)=1)
% 2. Evaluate FI at each grid point
% 3. Scale FI by prior probability of grid point


% ================================

% 1. Configure tissue paramater grid (depends on modeltype...)

% Vector of volume fractions for each volume fraction
fs = linspace(opts.fmin, opts.fmax, opts.nf);

% Volume fraction spacings
paramSpacings = ((opts.fmax-opts.fmin)/(opts.nf-1))*ones(Nparam,1);




switch modeltype

    case 'RDI v1.3' 

        % [fIC, fEES, R]

        % N dimensional grid for each parameter
        [varargout{1:Nparam-1}] = ndgrid(fs);

        % Make N dimenionsal parameter grid (last dimension is vector of volume fractions)
        paramGrid = cat(Nparam, varargout{1:Nparam-1});
        
        % Flatten grid
        paramGridFlat = reshape(paramGrid, opts.nf^(Nparam-1), (Nparam-1));
        
        % Only select elements with sum(f)=1
        paramGridFlat = paramGridFlat( sum(paramGridFlat,2) == 1, :);


end







% ================================

% 2. Evaluate FI at each grid point

% Initialise FI
FI = 0;

% Iterate over T2 values and tissue volume fractions
for T2 = opts.T2s

    for scanparamIndx = 1:size(paramGridFlat,1)
        

        switch modeltype

            case 'RDI v1.3'

                fIC = paramGridFlat(scanparamIndx,1);
                fEES = paramGridFlat(scanparamIndx,2);
                R = opts.R;

                % Get scan parameters
                tissue_params = [fIC, R, fEES];

        end


        % Evaluate fisher information
        fi = evaluateFisherInformation( ...
            modeltype,...
            paramIndx, ...
            scan_params, ...
            tissue_params, ...
            T2 = T2, ...
            TEconst = opts.TEconst, ...
            sigma = opts.sigma);


        % Add to FI
        FI = FI + fi;

    end

end