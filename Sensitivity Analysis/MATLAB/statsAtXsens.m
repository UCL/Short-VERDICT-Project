function scores = statsAtXsens(sens, scores, opts)


arguments
    sens % specified sensitivity
    scores % scores structure
    opts.Nboot = 1000;
    opts.CI = 95;


end


for protIndx  = 1:length(scores)

    % Read scores and labels (biopsy resuts)
    yscores = scores(protIndx).scores;
    ylabels = scores(protIndx).biopsyresults;
    
    % Create data array
    Data = [reshape(yscores, [length(yscores),1] ) , reshape(ylabels, [length(yscores),1] )];


    % Find threshold, stats, and sens at X sensitivity
    thresatXsens = ThresAtXSens(Data, sens);
    specatXsens = SpecAtXSens(Data, sens);
    sensatXsens = SensAtXSens(Data, sens);

    % Find confidence intervals
    BootThress = bootstrp(opts.Nboot, @ThresAtXSens, Data, sens);
    BootSpecs  = bootstrp(opts.Nboot, @SpecAtXSens, Data, sens);
    BootSenss = bootstrp(opts.Nboot, @SensAtXSens, Data, sens);

    % Find lower and upper CI thresholds
    lowthres = prctile(BootThress, (100-opts.CI)/2);
    highthres = prctile(BootThress, (100+opts.CI)/2);

    lowspec = prctile(BootSpecs, (100-opts.CI)/2);
    highspec = prctile(BootSpecs, (100+opts.CI)/2);

    lowsens = prctile(BootSenss, (100-opts.CI)/2);
    highsens = prctile(BootSenss, (100+opts.CI)/2);

    Thresholds = [thresatXsens, lowthres, highthres];
    Specificities = [specatXsens, lowspec, highspec];
    Sensitivities = [sensatXsens, lowsens, highsens];

    scores(protIndx).Thresholds = Thresholds;
    scores(protIndx).Sensitivities = Sensitivities;
    scores(protIndx).Specificities = Specificities;

end

end



function thres = ThresAt90Sens(Data)

% Function to take Data [yscore, ylabel] and output the Youden threshold

% First extract yscore and ylabel
yscore = Data(:,1);
ylabel = Data(:,2);

% Use MATLAB's perfcurve function
[X,Y,T] = perfcurve(ylabel ,yscore, 1);

% Sens and spec
senss = Y;
specs = 1-X;
thress = T;

% Find indx of near 90% sens
Indx = sum( senss < 0.9);

if abs( senss(Indx)-0.9 ) > abs( senss(Indx+1)-0.9 )
    Indx = Indx+1;
end

thres = thress(Indx);

end



function thres = ThresAtXSens(Data, sens)

% Function to take Data [yscore, ylabel] and output the Youden threshold

% First extract yscore and ylabel
yscore = Data(:,1);
ylabel = Data(:,2);

% Use MATLAB's perfcurve function
[X,Y,T] = perfcurve(ylabel ,yscore, 1);

% Sens and spec
senss = Y;
specs = 1-X;
thress = T;

% Find indx of near 90% sens
Indx = sum( senss < sens);

if abs( senss(Indx)-sens ) > abs( senss(Indx+1)-sens )
    Indx = Indx+1;
end

thres = thress(Indx);

end


function spec = SpecAtXSens(Data, sens)

% Function to take Data [yscore, ylabel] and output the Youden threshold

% First extract yscore and ylabel
yscore = Data(:,1);
ylabel = Data(:,2);

% Use MATLAB's perfcurve function
[X,Y,T] = perfcurve(ylabel ,yscore, 1);

% Sens and spec
senss = Y;
specs = 1-X;
thress = T;

% Find indx of near 90% sens
Indx = sum( senss < sens);

if abs( senss(Indx)-sens ) > abs( senss(Indx+1)-sens )
    Indx = Indx+1;
end

spec = specs(Indx);

end


function sens = SensAtXSens(Data, sens)

% Function to take Data [yscore, ylabel] and output the Youden threshold

% First extract yscore and ylabel
yscore = Data(:,1);
ylabel = Data(:,2);

% Use MATLAB's perfcurve function
[X,Y,T] = perfcurve(ylabel ,yscore, 1);

% Sens and spec
senss = Y;
specs = 1-X;
thress = T;

% Find indx of near 90% sens
Indx = sum( senss < sens);

if abs( senss(Indx)-sens ) > abs( senss(Indx+1)-sens )
    Indx = Indx+1;
end

sens = senss(Indx);

end