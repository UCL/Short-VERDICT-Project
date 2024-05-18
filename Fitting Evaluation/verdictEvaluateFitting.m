% Function to compare fIC fitting performance of different protocols

function [stats, figs] = verdictEvaluateFitting(modeltypes, schemenames, fittingtypes, opts)



arguments

    modeltypes % cell array of modeltypes (character vectors)
    schemenames  % cell array of scheme names
    fittingtypes % cell array of fitting techniques

    % ^lengths of cell arrays must match!

    % == Options

    % Folder where scheme structures are stored
    opts.schemesfolder = 'C:\Users\adam\OneDrive - University College London\UCL PhD\PhD Year 1\Projects\VERDICT Screening\Code\VERDICT-Screening\General Code\Model Fitting\MLP\My Tests\Schemes'
    
    % Fitting
    opts.Nrep = 10 % Number of fitting repetitions at each tissue parameter set
    opts.Niter = 40 % Number of iterations/parameter sets
    
    % Simulation
    opts.fICs = [0.1, 0.9] % min and max fIC values to use in simulation
    opts.fVASCs = [0, 0.1] % min and max fVASC values to use in simulation

    opts.simnoisetype = 'Rice'
    opts.T2 = 10000 % T2 value to use in simulation
    opts.NoiseSigma = 0.05 % Noise sigma for spin density image (can be single value of array of noises for each protocol)
    
    opts.Rs = linspace(0.1, 15.1, 17) % radii values to use in simulation
    opts.randtype = 'normal' % How fR distibution is randomised (uniform or Gaussian distribution)
    opts.muRs = [6,9] % mu and sigma ranges of radii distribution 
    opts.sigmaRs = [1,3]
    opts.TEconst = 22;

    % Fitting
    opts.NoisyMatrix = false

    % For MLP fitting
    opts.noisetype = 'Rice'
    opts.sigma0train = 0.05 % For MLP model (can be single value of array of noises for each protocol)
    opts.T2train = 10000

    % Saving figures
    opts.savefigs = true
    opts.figureFolder = "C:\Users\adam\OneDrive - University College London\UCL PhD\PhD Year 1\Projects\VERDICT Screening\Documents\Project Progress\Model Simplification\Figures"

end


%% If single protocol specified

if ischar(modeltypes)
    modeltypes = {modeltypes};
end

if ischar(schemenames)
    schemenames = {schemenames};
end

if ischar(fittingtypes)
    fittingtypes = {fittingtypes};
end





Nmodels = length(modeltypes);
Nschemes = length(schemenames);
Nfittypes = length(fittingtypes);
Nnoisesigma = length(opts.NoiseSigma);
Nsigma0train = length(opts.sigma0train);

Nprotocols = max([Nmodels, Nschemes, Nfittypes, Nnoisesigma, Nsigma0train]);

if Nmodels == 1
    modeltypes = repmat( modeltypes , 1, Nprotocols);
    Nmodels = length(modeltypes);
end

if Nschemes == 1
    schemenames = repmat( schemenames , 1, Nprotocols);
    Nschemes = length(schemenames);
end

if Nfittypes == 1
    fittingtypes = repmat( fittingtypes , 1, Nprotocols);
    Nfittypes = length(fittingtypes);
end

if length(opts.NoiseSigma) == 1
    opts.NoiseSigma = repmat(opts.NoiseSigma, 1, Nprotocols);
elseif length(opts.NoiseSigma) ~= Nprotocols
    error('Noise sigma array must be same length and number of protocols')
end

if length(opts.sigma0train) == 1
    opts.sigma0train = repmat(opts.sigma0train, 1, Nprotocols);
elseif length(opts.sigma0train) ~= Nprotocols
    error('sigma0train array must be same length and number of protocols')
end

% Check protocols well defined
if or((Nmodels~=Nschemes),(Nmodels~=Nfittypes))
    error('Cell arrays must be same length!')
end

Nprotocols = Nmodels;


% Initialise array for results
FittingBiases = zeros(Nprotocols, opts.Niter);
FittingVariances = zeros(Nprotocols, opts.Niter);

% Summary stats
stats = struct();


%% Evaluate fitting performance of each protocol

% Track simulated parameters
simfICs = zeros(opts.Niter,1);
simfRs = zeros(opts.Niter,length(opts.Rs));


% == Iterate over randomised tissue parameters

for iterIndx = 1:opts.Niter

    iterIndx
    
    % == Find randomised paramater set (ALWAYS SIMULATING WITH ORIGINAL
    % VERDICT MODEL)

    % fIC, fEES, fVASC
    fIC = opts.fICs(1) + ( opts.fICs(2) - opts.fICs(1) )*rand();

    if 1-fIC > opts.fVASCs(2)
        fVASC = opts.fVASCs(1) + (opts.fVASCs(2)-opts.fVASCs(1))*rand();
    else
        fVASC = opts.fVASCs(1) + (1-fIC-opts.fVASCs(1))*rand();
    end

    fEES = 1-fIC-fVASC;

    switch opts.randtype
    
        case 'uniform'

            % Radii distribution (FROM UNIFORM CURRENTLY)
            fRs = rand(length(opts.Rs), 1);

        case 'normal'
        
            % Radii distribution (Normal)
            muR = opts.muRs(1) + ( opts.muRs(2) - opts.muRs(1) )*rand();
            sigmaR = opts.sigmaRs(1) + ( opts.sigmaRs(2) - opts.sigmaRs(1) )*rand();
            fRs = normpdf(opts.Rs, muR, sigmaR);

    end

    % Scale by fIC
    fRs = (fIC/sum(fRs))*fRs;

    simfICs(iterIndx) = fIC;
    simfRs(iterIndx,:) = fRs;



    % == Evaluate fitting at this tissue parameter set for each protocol

    for protIndx = 1:Nprotocols

        modeltype = modeltypes{protIndx};
        schemename = schemenames{protIndx};
        fittingtype = fittingtypes{protIndx};

        % Load scheme
        load([opts.schemesfolder '/' schemename '.mat']);

      
        % Evaluate fitting
        [bias, variance] = verdict_evaluatefitting( ...
            scheme, ...
            schemename,...
            modeltype, ...
            fittingtype,...
            fIC = fIC,...
            fEES = fEES,...
            Rs = opts.Rs,...
            fRs = fRs,...
            T2 = opts.T2,...
            TEconst = opts.TEconst,...
            simnoisetype = opts.simnoisetype,...
            NoiseSigma = opts.NoiseSigma(protIndx),...
            Nrep = opts.Nrep,...
            NoisyMatrix = opts.NoisyMatrix,...
            noisetype=opts.noisetype,...
            sigma0train = opts.sigma0train(protIndx),...
            T2train = opts.T2train...
            );
    
    
        FittingBiases(protIndx, iterIndx) = bias;
        FittingVariances(protIndx, iterIndx) = variance;

    end

end


% == Summary stats

for protIndx = 1:Nprotocols
    % Mean bias
    stats.bias(protIndx) = mean( FittingBiases(protIndx, :) );
    % Mean variance
    stats.variance(protIndx) = mean( FittingVariances(protIndx, :) );
end



%% Figures

% Model colour dictionary
modelcolourdict = dictionary( ...
    {'Original VERDICT', 'No VASC VERDICT', 'RDI', 'RDI v1.3', 'RDI v1.4'}, ...
    [	"#0072BD", "#77AC30", "#9EA1D4","#77AC30", "#A8D1D1"]);


% == Make table of protocol information
Protocols = struct();

for protIndx = 1:Nprotocols

    Protocols(protIndx).Number = protIndx;
    Protocols(protIndx).ModelType = modeltypes{protIndx};
    Protocols(protIndx).SchemeName = schemenames{protIndx};
    Protocols(protIndx).FittingType = fittingtypes{protIndx};
    Protocols(protIndx).NoiseSigma = opts.NoiseSigma(protIndx);
    Protocols(protIndx).sigma0train = opts.sigma0train(protIndx);

end

protocolTable = struct2table(Protocols);


% == figure 1: fIC fitting bias

fig1 = figure;

% Scatter raw data
for protIndx = 1:Nprotocols
    modeltype = modeltypes{protIndx};
    scatter( ...
        protIndx*ones(opts.Niter,1), ...
        FittingBiases(protIndx, :), ...
        '*', MarkerEdgeColor = modelcolourdict({modeltype}), ...
        MarkerEdgeAlpha = 0.1,...
        HandleVisibility = 'off')
    hold on
end

scatter(-1,-1, '*', MarkerEdgeColor = modelcolourdict({'Original VERDICT'}), DisplayName = 'Full Model')
% scatter(-1,-1, '*', MarkerEdgeColor = modelcolourdict({'No VASC VERDICT'}), DisplayName = 'No VASC VERDICT')
% scatter(-1,-1, '*', MarkerEdgeColor = modelcolourdict({'RDI'}), DisplayName = 'RDI')
scatter(-1,-1, '*', MarkerEdgeColor = modelcolourdict({'RDI v1.3'}), DisplayName = 'No VASC')
% scatter(-1,-1, '*', MarkerEdgeColor = modelcolourdict({'RDI v1.4'}), DisplayName = 'RDI v1.4')
xlim([0,Nprotocols+1])
legend;

% Add boxplots
boxplot(transpose(FittingBiases))

xticks(linspace(1,Nprotocols,Nprotocols))
xticklabels(linspace(1,Nprotocols,Nprotocols))
xlabel('Protocol Number')

title('fIC fitting bias')



% == figure 2: fIC fitting variance

fig2 = figure;

% Scatter raw data
for protIndx = 1:Nprotocols
    modeltype = modeltypes{protIndx};
    scatter( ...
        protIndx*ones(opts.Niter,1), ...
        FittingVariances(protIndx, :), ...
        '*', MarkerEdgeColor = modelcolourdict({modeltype}), ...
        MarkerEdgeAlpha = 0.1,...
        HandleVisibility = 'off')
    hold on
end

scatter(-1,-1, '*', MarkerEdgeColor = modelcolourdict({'Original VERDICT'}), DisplayName = 'Full Model')
% scatter(-1,-1, '*', MarkerEdgeColor = modelcolourdict({'No VASC VERDICT'}), DisplayName = 'No VASC')
% scatter(-1,-1, '*', MarkerEdgeColor = modelcolourdict({'RDI'}), DisplayName = 'RDI')
scatter(-1,-1, '*', MarkerEdgeColor = modelcolourdict({'RDI v1.3'}), DisplayName = 'No VASC')
% scatter(-1,-1, '*', MarkerEdgeColor = modelcolourdict({'RDI v1.4'}), DisplayName = 'RDI v1.4')
xlim([0,Nprotocols+1])
legend;

% Add boxplots
boxplot(transpose(FittingVariances))

xticks(linspace(1,Nprotocols,Nprotocols))
xticklabels(linspace(1,Nprotocols,Nprotocols))
xlabel('Protocol Number')

title('fIC fitting variance')




% == figure 3: fitting bias dependence on fIC

fig3 = figure;

% Scatter plot
for protIndx = 1:Nprotocols

    modeltype = modeltypes{protIndx};

    subplot(1,Nprotocols, protIndx);
    scatter(simfICs, FittingBiases(protIndx, :),...
        '*', MarkerEdgeColor = modelcolourdict({modeltype}), ...
        MarkerEdgeAlpha = 0.1,...
        HandleVisibility = 'off')
    hold on
    ylim([-0.25,0.25])
    xlabel('Simulated fIC')
    ylabel(['Protocol ' num2str(protIndx) ': fIC fitting bias'])
    grid on

    % Regression
    X = ones(length(simfICs),2);
    X(:,2) = simfICs;
    Y = transpose(FittingBiases(protIndx, :));
    BetaVec = X\Y;

    x = linspace(0,1,2);
    y = BetaVec(2)*x+BetaVec(1);
    plot(x,y)
    hold on


end



% == figure 4: fitting variance dependence on fIC

fig4 = figure;

% Scatter plot
for protIndx = 1:Nprotocols

    modeltype = modeltypes{protIndx};

    subplot(1,Nprotocols, protIndx);
    scatter(simfICs, FittingVariances(protIndx, :),...
        '*', MarkerEdgeColor = modelcolourdict({modeltype}), ...
        MarkerEdgeAlpha = 0.1,...
        HandleVisibility = 'off')
    hold on
    ylim([0,0.06])
    xlabel('Simulated fIC')
    ylabel(['Protocol ' num2str(protIndx) ': fIC fitting variance'])
    grid on

    % Regression
    X = ones(length(simfICs),2);
    X(:,2) = simfICs;
    Y = transpose(FittingVariances(protIndx, :));
    BetaVec = X\Y;

    x = linspace(0,1,2);
    y = BetaVec(2)*x+BetaVec(1);
    plot(x,y)
    hold on


end


f = figure;
uit = uitable(f, 'Data', table2cell(protocolTable));
uit.ColumnName={protocolTable.Properties.VariableNames{:}};
uit.RowName=[];

% Saving figures
if opts.savefigs
    opts.figureFolder = char(opts.figureFolder);
    save([opts.figureFolder '/Protocols.mat'], 'protocolTable')
    saveas(f, [opts.figureFolder '/Protocols.png']);
    saveas(fig1, [opts.figureFolder '/fIC_bias.png'])
    saveas(fig2, [opts.figureFolder '/fIC_variance.png'])
    saveas(fig3, [opts.figureFolder '/bias_fIC_dependence.png'])
    saveas(fig4, [opts.figureFolder '/variance_fIC_dependence.png'])
end







end