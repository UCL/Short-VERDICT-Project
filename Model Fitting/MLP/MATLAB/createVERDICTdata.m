% MATLAB function to generate training data 

function [params, Signals] = createVERDICTdata(modeltype, schemename, opts)

arguments

    modeltype % specify model type
    schemename % specify scheme name

    % == OPTIONS

    opts.Ntrain = 100% Number of training samples

    
    % Noise
    opts.noisetype = 'Ratio'
    opts.sigma0 = 0.05 % 
    opts.T2 = 100 % (ms) 
    opts.TEconst = 22 % (ms) TE = delta + Delta + TEconst

    % Parameter ranges (restricted to better represent real tissue)
    opts.fICs = [0.1, 0.9]
    opts.fVASCs = [0, 0.2]

    % == VERDICT model
    opts.Rs = linspace(0.1,15.1,17)
    opts.randtype = 'normal'
    opts.randmuRs = [5,10] % range of fR distribution means
    opts.randsigmaRs = [1,3] % range of fR distribution sigmas

    % == RDI model
    opts.muRs = [5, 7.5, 10]
    opts.sigmaRs = [2,2,2]
    opts.muRrange = [5,10] % Range of R/muR for RDI v1.3 and v1.4
    opts.sigmaRrange = [1,3] % Range of sigmaR for RDI v1.4



    % Save scheme
    opts.savescheme = true;
    opts.schemeParentFolder = 'C:\Users\adam\OneDrive - University College London\UCL PhD\PhD Year 1\Projects\VERDICT Screening\Code\VERDICT-Screening\General Code\Model Fitting\MLP\My Tests\Schemes'

    % Save data
    opts.savedata = false
    opts.dataParentFolder = 'C:\Users\adam\OneDrive - University College London\UCL PhD\PhD Year 1\Projects\VERDICT Screening\Code\VERDICT-Screening\General Code\Model Fitting\MLP\My Tests\Training Data'

end

% STEPS

% 1. Build specified scheme
% 2. Create set of randomised model parameters (for specified model)
% 3. Simulate signal over scheme for each parameter set (distribution sampling)
% 4. Save training data

% =========================================================================

%% 1. Build scheme

switch schemename

    case 'Original Full'

        % Define scan paramaters
        V01 = [1,2,0, 54, 2000, 1];
        V1 = [3.5, 26.9, 90, 54, 2000, 3*6];

        V02 = [1,2,0, 68, 2000, 1];
        V2 = [10.5, 33.9, 500, 68, 2000, 3*6];

        V03 = [1,2,0, 94, 2000, 1];
        V3 = [23.5, 46.9, 1500, 94, 2000, 3*6];

        V04 = [1,2,0, 75, 4341, 1];
        V4 = [14.0, 37.4, 2000, 75, 4341, 3*6];

        V05 = [1,2,0, 87, 4108, 1];
        V5 = [20, 43.4, 3000, 87, 4108, 3*6];
        
        % Addition of b=0 and descending order!
        Vs = [...
            V05; V5;...
            V04; V4;...
            V03; V3;...
            V02; V2;...
            V01; V1...
            ];
        
        % build scheme
        scheme = BuildScheme(Vs);



%     case 'Original ex903000'
% 
%         % Define scan paramaters
%         V01 = [1,2,0, 54, 2000, 1];
%         V1 = [3.5, 26.9, 90, 54, 2000, 3];
% 
%         V02 = [1,2,0, 68, 2000, 1];
%         V2 = [10.5, 33.9, 500, 68, 2000, 6];
% 
%         V03 = [1,2,0, 94, 2000, 1];
%         V3 = [23.5, 46.9, 1500, 94, 2482, 9];
% 
%         V04 = [1,2,0, 75, 4341, 1];
%         V4 = [14.0, 37.4, 2000, 75, 4341, 9];
% 
%         V05 = [1,2,0, 87, 4108, 1];
%         V5 = [20, 43.4, 3000, 87, 4108, 9];
% 
% 
%         % Addition of b=0 and descending order!
%         Vs = [...
% %             V05; V5;...
%             V04; V4;...
%             V03; V3;...
%             V02; V2;...
% %             V01; V1...
%             ];
% 
%         % build scheme
%         scheme = BuildScheme(Vs);


    case 'Original ex905003000'

        % Define scan paramaters
        V01 = [1,2,0, 54, 2000, 1];
        V1 = [3.5, 26.9, 90, 54, 2000, 3*6];

        V02 = [1,2,0, 68, 2000, 1];
        V2 = [10.5, 33.9, 500, 68, 2000, 3*6];

        V03 = [1,2,0, 94, 2000, 1];
        V3 = [23.5, 46.9, 1500, 94, 2482, 3*6];

        V04 = [1,2,0, 75, 4341, 1];
        V4 = [14.0, 37.4, 2000, 75, 4341, 3*6];

        V05 = [1,2,0, 87, 4108, 1];
        V5 = [20, 43.4, 3000, 87, 4108, 3*6];

        
        % Addition of b=0 and descending order!
        Vs = [...
%             V05; V5;...
            V04; V4;...
            V03; V3;...
%             V02; V2;...
%             V01; V1...
            ];
        
        % build scheme
        scheme = BuildScheme(Vs);


%     case 'Original ex9020003000'
% 
%         % Define scan paramaters
%         V01 = [1,2,0, 54, 2000, 1];
%         V1 = [3.5, 26.9, 90, 54, 2000, 3];
% 
%         V02 = [1,2,0, 68, 2000, 1];
%         V2 = [10.5, 33.9, 500, 68, 2000, 6];
% 
%         V03 = [1,2,0, 94, 2000, 1];
%         V3 = [23.5, 46.9, 1500, 94, 2482, 9];
% 
%         V04 = [1,2,0, 75, 4341, 1];
%         V4 = [14.0, 37.4, 2000, 75, 4341, 9];
% 
%         V05 = [1,2,0, 87, 4108, 1];
%         V5 = [20, 43.4, 3000, 87, 4108, 9];
% 
%         % Addition of b=0 and descending order!
%         Vs = [...
% %             V05; V5;...
% %             V04; V4;...
%             V03; V3;...
%             V02; V2;...
% %             V01; V1...
%             ];
% 
%         % build scheme
%         scheme = BuildScheme(Vs);


    % case 'Adam Scheme v1.1'
    % 
    %     V01 = [1,2,0, 66, 4543, 1];
    %     V1 = [20,35, 2000, 70, 4000, 5];
    %     V02 = [1,2,0, 65, 3043,1];
    %     V2 = [10, 40, 1000, 65, 4000, 5];
    % 
    %     Vs = [...
    %         V01; V1;...
    %         V02; V2;...
    %         ];
    % 
    %     % build scheme
    %     scheme = BuildScheme(Vs);


        
    case 'Short Scheme v1'

        V01 = [1, 2, 0, 76, 4000, 1];
        V1  = [20, 41, 1800, 76, 4000, 3*6];
        V02 = [1, 2, 0, 62, 4000, 1];
        V2 = [12, 35, 1000, 62, 4000, 3*6];

        Vs = [...
            V01; V1;...
            V02; V2;...
            ];

        % build scheme
        scheme = BuildScheme(Vs);



    case 'Short Scheme NS'

        V01 = [1, 2, 0, 76, 4000, 1];
        V1  = [20, 41, 1800, 76, 4000, 3*3]; % Ndirections*avg high b
        V02 = [1, 2, 0, 62, 4000, 1];
        V2 = [12, 35, 1000, 62, 4000, 3*3];

        Vs = [...
            V01; V1;...
            V02; V2;...
            ];

        % build scheme
        scheme = BuildScheme(Vs);


    case 'NS Simulation Scheme'

        V00 = [1, 2, 0, 80, 4000, 1];
        V0 = [20, 40, 2500, 80, 4000, 3*3];
        V01 = [1, 2, 0, 76, 4000, 1];
        V1  = [20, 41, 1800, 76, 4000, 3*3]; % Ndirections*avg high b
        V02 = [1, 2, 0, 62, 4000, 1];
        V2 = [12, 35, 1000, 62, 4000, 3*3];


        Vs = [...
            V00; V0;...
            V01; V1;...
            V02; V2;...
            ];

        % build scheme
        scheme = BuildScheme(Vs);
end


if opts.savescheme
    save([opts.schemeParentFolder '\' schemename '.mat'], 'scheme')

end


%% 2. Randomise model paramaters

% == Specify number of parameters
switch modeltype

    case 'Original VERDICT'

        % Define radii
        Rs = opts.Rs;
        nR = length(Rs);

        % Define ncompart
        ncompart = 2;

        % Number of parameters
        nparam = nR + ncompart;

    case 'No VASC VERDICT'

        % Define radii
        Rs = opts.Rs;
        nR = length(Rs);

        % Define ncompart
        ncompart = 1;

        % Number of parameters
        nparam = nR + ncompart;


    case 'RDI'

        % Define radii distributions
        Rs = opts.Rs;
        muRs = opts.muRs;
        sigmaRs = opts.sigmaRs;
        
        % Number of distributions
        ndist = length(muRs);

        % Number of compartments
        ncompart = 1;

        % Number of parameters
        nparam = ndist + ncompart;

    case 'RDI v1.3'

        ncompart = 1;
        nparam = 2 + ncompart;

    case 'RDI v1.4'
        
        Rs = opts.Rs;
        ncompart = 1;
        nparam = 3 + ncompart;
        

end


% == Randomise parameters

% Initialise parameter array
params = zeros(opts.Ntrain, nparam); 

% Loop to fill array
for paramIndx = 1:opts.Ntrain

    % == Intracellular

%     % Randomise fIC
%     fIC = opts.fICs(1) + (opts.fICs(2)-opts.fICs(1))*rand();

    %% == INVESTIGATING FIC randomisation distribution!
    fICs = linspace(0,1,100);
    % pdf = ones(size(fICs));
    % pdf = 2*abs(1-2*fICs);
    pdf = (1/1.25)*(1 + 0.5*(cos(pi*fICs).^2));
%     pdf = (2/3)*(1 + sin(pi*fICs).^2);
    fIC = sampleDistribution(pdf, fICs);

    % number of IC compartments (induvidual radii or distributions)
    nIC = nparam - ncompart;


    switch modeltype
        
        case 'Original VERDICT'

            switch opts.randtype

                % fRs randomly generated from uniform distribution over Rs
                case 'uniform'

                    % uniform distribution
                    fs = rand(1,nIC);
                
                    % Normalise
                    fs = fIC*fs/sum(fs);
    
                
                % fRs generates as random uniform distribution
                case 'normal'
                    
                    randmuR = opts.randmuRs(1) + (opts.randmuRs(2) - opts.randmuRs(1))*rand();
                    randsigmaR = opts.randsigmaRs(1) + (opts.randsigmaRs(2) - opts.randsigmaRs(1))*rand();

                    fs = normpdf(Rs, randmuR, randsigmaR );

                    % Normalise
                    fs = fIC*fs/sum(fs);

            end


        case 'No VASC VERDICT'

            switch opts.randtype

                case 'uniform'
                    
                    % uniform distribution
                    fs = rand(1,nIC);
                
                    % Normalise
                    fs = fIC*fs/sum(fs);

                case 'normal'
                    
                    randmuR = opts.randmuRs(1) + (opts.randmuRs(2) - opts.randmuRs(1))*rand();
                    randsigmaR = opts.randsigmaRs(1) + (opts.randsigmaRs(2) - opts.randsigmaRs(1))*rand();

                    fs = normpdf(Rs, randmuR, randsigmaR );

                    % Normalise
                    fs = fIC*fs/sum(fs);

            end       


        case 'RDI'
    
            % uniform distribution
            fs = rand(1,nIC);
        
            % Normalise
            fs = fIC*fs/sum(fs); 

        case 'RDI v1.3'

            fIC = rand();
            R = opts.muRrange(1) + (opts.muRrange(2)-opts.muRrange(1))*rand();

            fs = [fIC, R];


        case 'RDI v1.4'

            fIC = rand();
            muR = opts.muRrange(1) + (opts.muRrange(2)-opts.muRrange(1))*rand();
            sigmaR = opts.sigmaRrange(1) + (opts.sigmaRrange(2)-opts.sigmaRrange(1))*rand();

            fs = [fIC, muR, sigmaR];

    end



    % == Remaining volume fractions

    % ncompart = 2
    if ncompart == 2

        if 1-fIC > opts.fVASCs(2)
            fVASC = opts.fVASCs(1) + (opts.fVASCs(2)-opts.fVASCs(1))*rand();
        else
            fVASC = opts.fVASCs(1) + (1-fIC-opts.fVASCs(1))*rand();
        end

        fEES = 1-fIC-fVASC;

        fend = [fEES, fVASC];
    
    % ncompart = 1    
    else
        fEES = 1-fIC;
        fVASC = 0;
        fend = fEES;
    end

    % == Append volume fractions to array
    params(paramIndx, :) = [fs, fend];


end



%% 3. Simulate signal over parameter sets

% Initialise signal array
Signals = zeros(opts.Ntrain, length(scheme));

switch modeltype

    case 'Original VERDICT'

        for paramIndx = 1:opts.Ntrain

            paramIndx/opts.Ntrain

            tps = params(paramIndx,:);
    
            % Simulate signal distributions
            signals = simulateSignalsOverScheme(...
            scheme,...
            modeltype,...
            fIC = sum(tps(1:end-2)),...
            fEES = tps(end-1),...
            fVASC = tps(end),...
            fRs = tps(1:end-2),...
            Rs = Rs,...
            sigma0 = opts.sigma0,...
            T2 = opts.T2,...
            TEconst = opts.TEconst,...
            noisetype=opts.noisetype...
            );

            Signals(paramIndx,:) = signals;


        end

    case 'No VASC VERDICT'

        for paramIndx = 1:opts.Ntrain

            paramIndx/opts.Ntrain

            tps = params(paramIndx,:);
    
            % Simulate signal distributions
            signals = simulateSignalsOverScheme(...
            scheme,...
            modeltype,...
            fIC = sum(tps(1:end-1)),...
            fEES = tps(end),...
            fVASC = 0,...
            fRs = tps(1:end-1),...
            Rs = Rs,...
            sigma0 = opts.sigma0,...
            T2 = opts.T2,...
            TEconst = opts.TEconst,...
            noisetype=opts.noisetype...
            );

            Signals(paramIndx,:) = signals;


        end

    case 'RDI'


        for paramIndx = 1:opts.Ntrain
    
            paramIndx/opts.Ntrain
    
            tps = params(paramIndx,:);
    
            % Simulate signal distributions
            signals = simulateSignalsOverScheme(...
            scheme,...
            modeltype,...
            fIC = sum(tps(1:end-1)),...
            fEES = tps(end),...
            fVASC = 0,...
            fRdists = tps(1:end-1),...
            Rs = Rs,...
            muRs = muRs,...
            sigmaRs = sigmaRs,...
            sigma0 = opts.sigma0,...
            T2 = opts.T2,...
            TEconst = opts.TEconst,...
            noisetype=opts.noisetype...
            );
    
            Signals(paramIndx,:) = signals;
    
        end


    case 'RDI v1.3'


        for paramIndx = 1:opts.Ntrain
    
            paramIndx/opts.Ntrain
    
            tps = params(paramIndx,:);
    
            % Simulate signal distributions 
            signals = simulateSignalsOverScheme(...
            scheme,...
            modeltype,...
            fIC = tps(1),...
            fEES = tps(end),...
            fVASC = 0,...
            muR = tps(2),...
            sigma0 = opts.sigma0,...
            T2 = opts.T2,...
            TEconst = opts.TEconst,...
            noisetype=opts.noisetype...
            );
    
            Signals(paramIndx,:) = signals;
    
        end


    case 'RDI v1.4'


        for paramIndx = 1:opts.Ntrain
    
            paramIndx/opts.Ntrain
    
            tps = params(paramIndx,:);
    
            % Simulate signal distributions
            signals = simulateSignalsOverScheme(...
            scheme,...
            modeltype,...
            fIC = tps(1),...
            fEES = tps(end),...
            fVASC = 0,...
            muR = tps(2),...
            sigmaR = tps(3),...
            Rs = Rs,...
            sigma0 = opts.sigma0,...
            T2 = opts.T2,...
            TEconst = opts.TEconst,...
            noisetype = opts.noisetype...
            );
    
            Signals(paramIndx,:) = signals;
    
        end

end



%% 4. Save training data
if opts.savedata

    % Define output folder
    outputFolder = join([opts.dataParentFolder '/' modeltype '/' schemename '/' opts.noisetype '/T2_' num2str(opts.T2) '/sigma_' num2str(opts.sigma0)], "");
    
    if ~exist(outputFolder, "dir")
       mkdir(outputFolder)
    end

    save([outputFolder '/params.mat'], 'params');
    save([outputFolder '/signals.mat'], 'Signals');

    % Save META data structure
    Meta = struct();
    Meta.DateTime = datetime();
    Meta.Ntrain = opts.Ntrain;
    Meta.sigma0 = opts.sigma0;
    Meta.T2 = opts.T2;
    Meta.TEconst = opts.TEconst;
    
    Meta.modeltype = modeltype;
    Meta.scheme = scheme;
    Meta.schemename = schemename;
    Meta.Rs = opts.Rs;
    Meta.randtype = opts.randtype;
    Meta.fICpdf = pdf;
    Meta.randmuRs = opts.randmuRs;
    Meta.randsigmaRs = opts.randsigmaRs; 
    Meta.muRs = opts.muRs;
    Meta.sigmaRs = opts.sigmaRs;
    Meta.noisetype = opts.noisetype;

    save([outputFolder '/Meta.mat'], 'Meta');

end

