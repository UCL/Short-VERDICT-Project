% Testing bland altman analysis function

% Read list of patient numbers with ROIs
PatNums = dir("C:\Users\adam\OneDrive - University College London\UCL PhD\PhD Year 1\INNOVATE Data\ROIs\NT\DICOM");
PatNums = {PatNums.name};
PatNums = PatNums(3:end);

% % Exclude
% exclude = {'INN_291', 'INN_155', 'INN_167'};
% 
% for pat = exclude
%     where = (string(PatNums) == pat);
%     PatNums(where)=[];
% end

% Protocol names
protocolnames = {'Protocol 3', 'Protocol 4'};

% Model types
modeltypes = {'Original VERDICT', 'No VASC VERDICT'};

% Scheme names
schemenames = {'Original ex905003000', 'Original ex905003000'};

% Fitting techniques 
fittingtypes = {'MLP', 'MLP'};

% sigma0 train
sigma0trains = {0.05};


[fig, stats, protocols] = verdictBlandAltman(PatNums, modeltypes, schemenames, fittingtypes, sigma0train = sigma0trains, protocolnames = protocolnames);