% Function to construct fitting matrix

function A = constructMatrix(scheme, params)

arguments

    scheme % Diffusion imaging scheme (scan parameters)

    % Fitting paramaters 

    % (for verdict fitting)
    params.fitting_type = 'verdict' % ('verdict for fitting to induvidual radii, RDI for distribution fitting')
    params.ncompart = 2 % number of compartments 
    params.Rmin = 0.1
    params.Rmax = 15.1
    params.nR = 17 % Number of radii in original VERDICT fitting
    params.DIC = 2;
    params.DEES = 2;
    params.DVASC = 8;

    % (for RDI fitting)
    params.muRs = [6, 7.5, 9]
    params.sigmaRs = [2,2,2]
    params.DEESs = [2]

    % Other
    params.NoisyMatrix = false;
    params.NoiseSigma = 0.05;
    params.regulariser = 'Tikhonov'
    params.lambda = sqrt(0.001);
end


% Odds n ends
nscheme = length(scheme);

switch params.fitting_type

    case 'verdict'

        % Construct Rs
        Rs = linspace(params.Rmin, params.Rmax, params.nR);

        % Number of radii
        nr = params.nR;

        % Number of compartments
        ncompart = params.ncompart;

        % Group params
        t.Rs = Rs;
        t.dIC = params.DIC;
        t.dEES = params.DEES;
        t.dVASC = params.DVASC;


        %% Construct Matrix

        nUnknown = nr+ncompart;
        
        A = zeros([nscheme nr+ncompart]) ; 
        sIC  = zeros([nscheme nr]) ;
        sEES = zeros([1 nscheme]) ;
        sVASC = zeros([1 nscheme]) ;
    
        for ischeme = 1:nscheme
        
            % == Sphere signals
            for ir = 1:nr
                rsignal = sphereGPD(scheme(ischeme).delta, scheme(ischeme).DELTA, ...
                   scheme(ischeme).G, t.Rs(ir), t.dIC);
        
                % Add noise average
                if params.NoisyMatrix  && (scheme(ischeme).bval ~= 0)
                    rsignal = RicianNoiseAverage(rsignal, params.NoiseSigma);
                end
        
                sIC(ischeme,ir) = rsignal;
            end
        
        
            % == EES signal
            EESsignal = ball(scheme(ischeme).bval, t.dEES) ;
            % Add noise average
            if (params.NoisyMatrix) && (scheme(ischeme).bval ~= 0)
                disp(EESsignal)
                EESsignal = RicianNoiseAverage(EESsignal, params.NoiseSigma);
                disp(EESsignal)
            end
            sEES(ischeme)  = EESsignal;
        
        
            % == VASC signal
            VASCsignal = astrosticks(scheme(ischeme).bval, t.dVASC) ;
            % Add noise average
            if params.NoisyMatrix && (scheme(ischeme).bval ~= 0)
                VASCsignal = RicianNoiseAverage(VASCsignal, params.NoiseSigma);
            end
            sVASC(ischeme)  = VASCsignal;
        
        
            if ncompart == 2
                A(ischeme,:)   = [sIC(ischeme,:) sEES(ischeme) sVASC(ischeme) ] ;
            else
                A(ischeme,:)   = [sIC(ischeme,:) sEES(ischeme)] ;
            end
        end



    case 'RDI'

        % Construct Rs
        nR = 50;
        Rs = linspace(params.Rmin, params.Rmax, nR);

        % Construct radii distributions
        nRDists = length(params.muRs);
        
        normfRs = zeros(nRDists, nR);
        
        for RDistIndx = 1:nRDists
            
            muR = params.muRs(RDistIndx);
            sigmaR = params.sigmaRs(RDistIndx);
            fRs = normpdf(Rs, muR, sigmaR);
            normfRs(RDistIndx,:) = fRs/sum(fRs);
        
        end


        % Number of diffusivities
        nDEESs = size(params.DEESs,2);
        DEESs = params.DEESs;

        % group params
        t.dIC = params.DIC;
        
        %% Construct matrix
        
        % Number of unknowns
        nUnknown = nRDists + nDEESs;
        
        A = zeros([nscheme nUnknown]);
        sIC = zeros([nscheme nRDists]);
        sEES = zeros([nscheme nDEESs]);
        
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
                if params.NoisyMatrix && (scheme(ischeme).bval ~= 0)
                    RDistSignal = RicianNoiseAverage(RDistSignal, params.NoiseSigma);
                end
        
                ICsignal(1,RDistIndx) = RDistSignal; 
        
            end
        
        
            sIC(ischeme, :) = ICsignal;
        
        
            % === EES signal
        
            EESsignal  = zeros(1, nDEESs);
        
            % Loop over diffusivities
            for Dindx = 1:nDEESs
        
                signal = ball(scheme(ischeme).bval, DEESs(1,Dindx));
            
                % Noise
                if (params.NoisyMatrix) && (scheme(ischeme).bval ~= 0)
                    signal = RicianNoiseAverage(signal, params.NoiseSigma);
                end
        
                EESsignal(1,Dindx) = signal;
        
            end
        
            sEES(ischeme, :)  = EESsignal;
        
          
        
            A(ischeme,:) = [sIC(ischeme,:) sEES(ischeme,:)];
        
        end

end
% 

% Augmented matrix for regularisation

switch params.regulariser

    case 'Tikhonov'

        L = eye(nUnknown);
            
        A = [A; params.lambda*L];

end

end