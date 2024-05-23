function FI = evaluateFisherInformation(modeltype, paramIndx, scan_params, tissue_params, opts)

arguments
    modeltype
    paramIndx % Model parameter to evaluate Fisher Information 
    scan_params % Vector of scan parameters [delta, Delta, b]
    tissue_params % Vector of tissue volume fractions 
    
    % verdict: [fR1, fR2, ..., fEES, fVASC]
    % no VASC: [fR1, fR2, ..., fEES]
    % RDI: [fRdist1, fRdist2, ..., fEES]

    % == Options
    opts.T2 = 100 % T2 value
    opts.TEconst = 22 % TE = delta + Delta + TEconst 
    opts.sigma = 0.05 % Noise level
    opts.paramstepsize = 0.01 % step size when calculating partial derivatives

end


% Function to evaluate Fisher information of a specified set of sacn
% parameters, for a specified set of tissue paramaters

% STEPS

% 1. Evaluate signal distribution at tissue parameters
% 2. Evaluate signal distribution at tissue_parameters +- dTheta(paramIndx)
% 3. Evaluate logarithms of signal distributions
% 4. Evaluate second order partial derivative of log signal distribution with respect to
% specified model parameter
% 5. Evaluate expected value of second order partial derivative

% =====================================================

% == Step 1. Evaluate signal distribution at tissue parameters

[signalDist, signals] = signalDistribution( ...
    scan_params, ...
    tissue_params, ...
    modeltype = modeltype,...
    T2 = opts.T2, ...
    TEconst = opts.TEconst);

signalDist = signalDist + eps;

% == Step 2: Evaluate signal distribution at tissue_parameters +- dTheta(paramIndx)

% define parameter step size
dparam = opts.paramstepsize;

% Define step up and down params (case dependent)
paramsUp = tissue_params;
paramsDown = tissue_params;

if tissue_params(paramIndx) == 0

    paramsUp(paramIndx) = paramsUp(paramIndx) + dparam;
    deltaparam = dparam;

elseif tissue_params(paramIndx) == 1

    paramsDown(paramIndx) = paramsDown(paramIndx) - dparam;
    deltaparam = dparam;

else
    paramsUp(paramIndx) = paramsUp(paramIndx) + dparam;
    paramsDown(paramIndx) = paramsDown(paramIndx) - dparam;
    deltaparam = 2*dparam;

end

% Normalise volume fraction parameters

switch modeltype

    case 'RDI v1.3'

        UpSum = paramsUp(1)+paramsUp(3);
        paramsUp(1) = (1/(UpSum))*paramsUp(1);
        paramsUp(3) = (1/(UpSum))*paramsUp(3);

        DownSum = paramsDown(1)+paramsDown(3);
        paramsDown(1) = (1/(DownSum))*paramsDown(1);
        paramsDown(3) = (1/(DownSum))*paramsDown(3);


end
% Evaluate signal distributions (step up and step down)

signalDistUp = signalDistribution( ...
    scan_params, ...
    paramsUp, ...
    modeltype = modeltype,...
    T2 = opts.T2, ...
    TEconst = opts.TEconst) + eps;

signalDistDown = signalDistribution( ...
    scan_params, ...
    paramsDown, ...
    modeltype = modeltype,...
    T2 = opts.T2, ...
    TEconst = opts.TEconst) + eps;


% == Step 3. Evaluate logarithm of signal distributions

logsignalDist = log(signalDist);
logsignalDistUp = log(signalDistUp);
logsignalDistDown = log(signalDistDown);

% == Step 4. Evaluate first order partial derivative
% 
% % u'' = (1/h^2)*(u(x+h)-2u(x)+u(x-h))
% logsignalDist_partial2 = (1/deltaparam^2)*(logsignalDistUp - 2*logsignalDist + logsignalDistDown);
% % Remove NaN
% logsignalDist_partial2(isnan(logsignalDist_partial2)) = 0;

% u' = (1/2h)*(u(x+h)-u(x-h))
logsignalDist_partial1 = (1/deltaparam)*(logsignalDistUp - logsignalDistDown);

% partial derivative squared
logsignalDist_partial2 = (logsignalDist_partial1.^2);

% == Step 5. Evaluate expected value of log second partial derivative

Expt_logpartial2 = trapz( logsignalDist_partial2.*signalDist);

FI = Expt_logpartial2;



end