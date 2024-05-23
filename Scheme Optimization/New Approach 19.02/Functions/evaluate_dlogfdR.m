function experiment = evaluate_dlogfdR(experiment)

arguments
    % Structure containing experiment detail
    experiment
end

% Function evaluates dfdR for each scan in scheme, for each element of the
% prior grid.

scheme = experiment.scheme;
priorgrid = experiment.priorgrid;


% Loop over scans/scheme elements
for scanIndx = 1:length(scheme)

    % Read scan params
    scan_params = [
        scheme(scanIndx).delta,...
        scheme(scanIndx).DELTA,...
        scheme(scanIndx).bval,...
        scheme(scanIndx).TE,...
        scheme(scanIndx).TR,...
        scheme(scanIndx).Nav_ratio
        ];
    
    % Loop over prior grid of tissue parameters
    for paramsIndx = 1:size(priorgrid,1)

        fIC = priorgrid(paramsIndx, 1);
        R = priorgrid(paramsIndx, 2);
        fEES = 1 - fIC;

        tissue_params = [fIC, R, fEES];

        % define parameter step size
        dR = experiment.options.Rstepsize;
        
        % Define step up and down params (case dependent)
        paramsUp = tissue_params;
        paramsDown = tissue_params;
        
        if tissue_params(2) == 0
        
            paramsUp(2) = paramsUp(2) + dR;
            deltaparam = dparam;
        
        elseif tissue_params(2) == 1
        
            paramsDown(2) = paramsDown(2) - dR;
            deltaparam = dR;
        
        else
            paramsUp(2) = paramsUp(2) + dR;
            paramsDown(2) = paramsDown(2) - dR;
            deltaparam = 2*dR;
        
        end


        signalDistUp = signalDistribution( ...
        scan_params, ...
        paramsUp, ...
        modeltype = experiment.modeltype,...
        T2 = experiment.T2,...
        sigma = experiment.sigma0) + eps;
        
        signalDistDown = signalDistribution( ...
            scan_params, ...
            paramsDown, ...
            modeltype = experiment.modeltype,...
            T2 = experiment.T2,...
            sigma = experiment.sigma0) + eps;

        % == Evaluate logarithm of signal distributions

        logsignalDistUp = log(signalDistUp);
        logsignalDistDown = log(signalDistDown);
        
        % == Evaluate first order partial derivative
        % u' = (1/2h)*(u(x+h)-u(x-h))
        logsignalDist_partial1 = (1/deltaparam)*(logsignalDistUp - logsignalDistDown);

        if and(scanIndx == 1, paramsIndx == 1)
            experiment.dlogfdRs = zeros( length(scheme), length(priorgrid), length(signalDistUp));
        end

        experiment.dlogfdRs(scanIndx, paramsIndx, :) = logsignalDist_partial1;


    end


end