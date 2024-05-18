function [fig, stats] = myBlandAltman(Data1, Data2, opts)

% Function to construct Bland Altman plot from input data

arguments
    Data1 % Array of first data set
    Data2 % Array of second data set
    opts.Data1Label = 'Data 1' % Label for data 1 array
    opts.Data2Label = 'Data 2'% Label for data 2 array
    opts.MarkerAlpha = 0.3
    opts.xlim = []
    opts.ylim = []
end


% First, check lengths of data 1 and data2 are the same
if length(Data1) == length(Data2)
    disp('')
else
    error("Data sizes don't match!")
end

disp(length(Data1))
% Next construct arrays of mean and differences
means = 0.5*(Data1+Data2);
diffs = (Data1-Data2);

% Calculate mean and std of differences
mu_diffs = mean(diffs);
sigma_diffs = std(diffs);

% Calculate mean standard error
muSE = sigma_diffs/sqrt(length(Data1));

% Caclualte limits of agreement
ULoA = mu_diffs+1.96*sigma_diffs;
LLoA = mu_diffs-1.96*sigma_diffs;

% Construct figure
fig = figure;

% scatter of means and diffs
scatter(means, diffs, '*', 'MarkerEdgeAlpha', opts.MarkerAlpha)
hold on
% horizontal lines for mu_diff and limits of agreements
yline(mu_diffs )
text(1*max(means), mu_diffs, num2str(round(mu_diffs,4), '%4.3f'), 'HorizontalAlignment','left','VerticalAlignment','bottom')
hold on
yline(mu_diffs+1.96*sigma_diffs)
text(0.85*max(means), mu_diffs+1.96*sigma_diffs, [num2str( round(mu_diffs+1.96*sigma_diffs,4), '%4.3f') ' (+1.96 SD)'], 'HorizontalAlignment','left','VerticalAlignment','bottom')
hold on
yline(mu_diffs-1.96*sigma_diffs)
text(0.85*max(means), mu_diffs-1.96*sigma_diffs, [num2str(round(mu_diffs-1.96*sigma_diffs,4), '%4.3f') ' (-1.96 SD)'], 'HorizontalAlignment','left','VerticalAlignment','bottom')
hold on

% Label axes
ylabel([opts.Data1Label ' - ' opts.Data2Label])
xlabel(['Mean of ' opts.Data1Label ' and ' opts.Data2Label])

% Set limits
if isempty(opts.ylim)
    ylim([mu_diffs-5*sigma_diffs, mu_diffs+5*sigma_diffs])
else
    ylim(opts.ylim)
end

if isempty(opts.xlim)
    xlim([min(means), max(means)])
else
    xlim(opts.xlim)
end

grid on

stats = [mu_diffs, sigma_diffs, ULoA, LLoA, muSE];

end