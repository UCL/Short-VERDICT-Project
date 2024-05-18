function [fIC, fEES, fVASC, R, rmse] = RDI_fit(scheme,Y,t,opt)

% Perform Restricted Diffusion Imaging (RDI) fit for fIC and fEES
% 
% [fIC, fEES] = RDI_fit(scheme, Y, t, opt)
% 
% RDI fitting assumes normal cellular radii distribution as seen from histology 
% patches and only has two fitting parameters fIC and fEES
%     -> muR = 8µm
%     -> sigmaR = 2µm

% L2 Tikonhov regularisation used to penalise extreme measurements


arguments 
    scheme (1,:)  struct
    Y {mustBeNumeric,mustBeReal}  % [ny nx nmeas] or [ny nx nz nmeas] 

    t.dEES  (1,1) {mustBeNumeric,mustBeReal} = 2
    t.dIC   (1,1) {mustBeNumeric,mustBeReal} = 2

    opt.muRs = [5, 7.5, 10];
    opt.sigmaRs = [2, 2, 2];
    opt.mask {mustBeNumeric,mustBeReal} = []
    opt.ncompart = 1;
    opt.solver = 'lsqnonnegTikhonov'
    opt.NoisyMatrix = false;
    opt.NoiseSigma = 0.05;


end

% Scheme, Data, and Image sizes
nscheme = length(scheme) ;
szY = size(Y) ;
szmap = szY(1:end-1) ;


if ~exist('opt','var') || isempty(opt.mask)
    opt.mask = ones(szmap) ;
end

if szY(end) ~= nscheme
    warning('MATLAB:verdict_fit:dataSizeInconsistencies', ...
        ['Number of measures (',num2str(szY(end)), ...
        ') must be same as size of schemes (', num2str(nscheme),')'])
end





%% Construct Matrix


% == Define radii distributions
nR = 50;
minR = 0.1;
maxR = 15.1;
Rs = linspace(minR, maxR, nR);

muRs = opt.muRs;
sigmaRs = opt.sigmaRs;

nRDists = length(muRs);

normfRs = zeros(nRDists, nR);

for RDistIndx = 1:length(muRs)
    
    muR = muRs(RDistIndx);
    sigmaR = sigmaRs(RDistIndx);
    fRs = normpdf(Rs, muR, sigmaR);
    normfRs(RDistIndx,:) = fRs/sum(fRs);

end


% === Define extracellular diffusivities
EES_Ds = [t.dEES]; % µm^2/ms

% Number of diffusivities
nEES_Ds = size(EES_Ds,2);

% === Construct matrix

% Number of unknowns
nUnknown = nRDists + nEES_Ds;

A = zeros([nscheme nUnknown]);
sIC = zeros([nscheme nRDists]);
sEES = zeros([nscheme nEES_Ds]);


% Simulate signal

for ischeme = 1:nscheme

    % === IC signal

    ICsignal = zeros(1,nRDists);

    for RDistIndx = 1:nRDists

        RDistSignal = 0;

        for Rindx = 1:nR
    
            Rsignal = normfRs(RDistIndx, Rindx)*sphereGPD(scheme(ischeme).delta, scheme(ischeme).DELTA, ...
               scheme(ischeme).G, Rs(Rindx), t.dIC);
    
            RDistSignal = RDistSignal + Rsignal;
    
        end

        % Noise 
        if opt.NoisyMatrix && (scheme(ischeme).bval ~= 0)
            RDistSignal = RicianNoiseAverage(RDistSignal, opt.NoiseSigma);
        end

        ICsignal(1,RDistIndx) = RDistSignal; 

    end


    sIC(ischeme, :) = ICsignal;


    % === EES signal

    EESsignal  = zeros(1, nEES_Ds);

    % Loop over diffusivities
    for Dindx = 1:nEES_Ds

        signal = ball(scheme(ischeme).bval, EES_Ds(1,Dindx));
    
        % Noise
        if (opt.NoisyMatrix) && (scheme(ischeme).bval ~= 0)
            signal = RicianNoiseAverage(signal, opt.NoiseSigma);
        end

        EESsignal(1,Dindx) = signal;

    end

    sEES(ischeme, :)  = EESsignal;

  

    A(ischeme,:) = [sIC(ischeme,:) sEES(ischeme,:)];

end


%% Fitting

fIC = zeros(szmap);
fEES = zeros(szmap);
rmse = zeros(szmap);
R = zeros(szmap);

% Flatten
opt.mask = opt.mask(:) ;
fIC = fIC(:); fEES = fEES(:);
Y = reshape(Y,[prod(szmap) nscheme]) ;



locinmask = find(opt.mask) ;

for ip = 1:length(locinmask)  % can be parallelised

    
    
    ind = locinmask(ip) ;
    y = Y(ind,:) ;

    percent_done = (ip/length(locinmask))*100
    
    % Remove infinties and NaN
    y(y==inf) = 0;
    y(y==-inf) = 0;
    y(isnan(y)) = 0;


    switch opt.solver

        case 'lsqnonnegTikhonov'
    
            nUnknown = size(A,2) ;
            lambda = sqrt(0.001) ;
   
            L = eye(nUnknown) ;
    
            AT = [A; lambda*L] ;
            YT = [y(:) ; zeros([nUnknown 1])] ;

            x = lsqnonneg(AT, YT) ;

%         case 'lsqnonnegSparse'
% 
%             nUnknown = size(A,2) ;
%             lambda = sqrt(0.001) ;
%     
%     
%             L = zeros(nUnknown) ;
%             L(1,1:nRDists) = 1;
%             L(2:nRDists+1:end) = 1;
%     
%             AT = [A; lambda*L] ;
%             YT = [y(:) ; zeros([nUnknown 1])] ;
% 
%             x = lsqnonneg(AT, YT) ;

    end

    fIC(ind) = sum( x(1:nRDists) );
    R(ind) = (muRs*x(1:nRDists))/fIC(ind);
    fEES(ind) = sum(x(nRDists+1:end) );
    rmse(ind) = norm(y(:) - A*x) / sqrt(nscheme) ;

  
end

fIC = reshape(fIC,szmap) ;
fEES = reshape(fEES,szmap) ;
fVASC = zeros(size(fIC));
rmse = reshape(rmse, szmap) ;

end

