% MATLAB script to run VERDICT processing on specified patients

%% Define study path (path to folder containing patients)
% 
% INNOVATE
STUDY_path = "C:\Users\adam\OneDrive\Desktop\INNOVATE STUDY COHORT VERDICT IMAGES";

% % Patient volunteers (Full VERDICT)
% STUDY_path = "C:\Users\adam\OneDrive - University College London\UCL PhD\PhD Year 1\Projects\Short VERDICT Project\Imaging Data\Patient Volunteers\Original VERDICT";

% % Patient volunteers (Short VERDICT)
% STUDY_path = "C:\Users\adam\OneDrive - University College London\UCL PhD\PhD Year 1\Projects\Short VERDICT Project\Imaging Data\Patient Volunteers\Short VERDICT";

% 
% % MARTA
% STUDY_path = "C:\Users\adam\OneDrive\Desktop\Marta\AP Organised";

%% Define patient numbers

% FOR VOLUNTEERS
PatNums = {"BAR_003"};%"HMU_066";"HMU_069"; "HMU_076"; "HMU_084"; "HMU_121"; "HMU_176"; "HMU_180"; "HMU_201"};

% % FOR INNOVATE
% x = dir(STUDY_path);
% PatNums = {x(:).name};
% PatNums = transpose(PatNums(3:end));

%% DEFINE FOLDERS

% Define output folder
OutputFolder = string(fileread("C:\Users\adam\OneDrive - University College London\UCL PhD\PhD Year 1\Projects\Short VERDICT Project\Code\Short-VERDICT-Project\output_folder.txt"));
% OutputFolder= "C:\Users\adam\OneDrive\Desktop\Marta\AP Organised\VERDICT OUTPUTS";


% Define schemes folder
schemesfolder = "C:\Users\adam\OneDrive - University College London\UCL PhD\PhD Year 1\Projects\Short VERDICT Project\Code\Short-VERDICT-Project\Model Fitting\Schemes";

% Define models folder
modelsfolder = "C:\Users\adam\OneDrive - University College London\UCL PhD\PhD Year 1\Projects\Short VERDICT Project\Code\Short-VERDICT-Project\Model Fitting\MLP\MLP Models";

% Define python folder
pythonfolder = "C:\Users\adam\OneDrive - University College London\UCL PhD\PhD Year 1\Projects\Short VERDICT Project\Code\Short-VERDICT-Project\Model Fitting\MLP\Python";


%% DEFINE VERDICT PROTOCOL

% === Model type
modeltype = 'Original VERDICT'; 

% === Scheme name
schemename = 'Original Full';

% === fitting technique
fittingtechnique =   'MLP';

% === Noise used in MLP training
noisetype='Rice';
sigma0train = 0.05;
T2train = 10000;

% === ADC
calcADC = true;
% Max b value
vADCbmax = 1501;


% Run VERDICT processing
for indx = 1:size(PatNums,1)
    PatNum = PatNums{indx};
    disp(["----------->" PatNums(indx)])
    VERDICT( ...
        PatNum, ...
        modeltype = modeltype,...
        schemename = schemename,...
        fittingtechnique = fittingtechnique,...
        noisetype=noisetype,...
        sigma0train = sigma0train,...
        T2train=T2train,...
        STUDY_path=STUDY_path,...
        parent_folder=OutputFolder,...
        schemesfolder = schemesfolder,...
        modelsfolder = modelsfolder,...
        pythonfolder = pythonfolder,...
        calcADC = calcADC,...
        vADCbmax = vADCbmax...
        );
end




