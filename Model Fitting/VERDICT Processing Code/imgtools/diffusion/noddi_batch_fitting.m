function noddi_batch_fitting(roi, bval, bvec, outputfile, sz_data, mask_locs, poolsize)
% NODDI_BATCH_FITTING Calls NODDI fitting routines
%  noddi_batch_fitting(vol_out, bval, bvec, outputfile, sz_data, mask_locs, poolsize))
%
%
% Adapted from code by Gary Hui Zhang (gary.zhang@ucl.ac.uk)
%
% See Also DICOM2NODDI NODDI_PARAMS_VIEW
%

if ~exist('MakeModel')
    disp([' You need the NODDI MATLAB Toolbox on your path.'])
    return
end

model = MakeModel('WatsonSHStickTortIsoV_B0');
protocol = noddi2Protocol(bval, bvec) ;

% first check if there is a file there to resume
if exist(outputfile, 'file')
    load(outputfile);
    if exist('split_end', 'var')
        % previously run and need to be restarted
        current_split_start = split_end + 1;
        fprintf('Resume an interrupted run from %i\n', current_split_start);
    else
        % completed
        fprintf('An output file of the same name detected.\n');
        fprintf('Choose a different output file name.\n');
        return;
    end
else
    % if this is the first run
    current_split_start = 1;
end

% initiate the parallel environment if necessary
if	matlabpool('size')==0
    if (nargin < 7)
        matlabpool
    else
        matlabpool('OPEN', poolsize);     
    end
end


numOfVoxels = size(roi,1);

% set up the fitting parameter variables if it is the first run
if current_split_start == 1
    gsps = zeros(numOfVoxels, model.numParams);
    mlps = zeros(numOfVoxels, model.numParams);
    fobj_gs = zeros(numOfVoxels, 1);
    fobj_ml = zeros(numOfVoxels, 1);
    error_code = zeros(numOfVoxels, 1);
    if model.noOfStages == 3
        mcmcps = zeros(numOfVoxels, model.MCMC.samples, model.numParams + 1);
    end
end

% set up the PARFOR Progress Monitor
% [mypath myname myext] = fileparts(mfilename('fullpath'));
% mypath = [mypath '/../ParforProgMonv2/java'];
% pctRunOnAll(['javaaddpath '  mypath]);
 progressStepSize = 1000;
% ppm = ParforProgMon(['Fitting ' roifile, ' : '], numOfVoxels-current_split_start+1,...
%                     progressStepSize, 400, 80);

tic

fprintf('%i of voxels to fit\n', numOfVoxels-current_split_start+1);
ss = [current_split_start:progressStepSize:numOfVoxels] ;
nss = length(ss) ;
disp(['Number of parallel loops will be:',num2str(nss)])
iss = 0 ;
% start the parallel fitting
for split_start=current_split_start:progressStepSize:numOfVoxels
    iss = iss+ 1;
    disp(['Progress: ',num2str( iss/nss*100)])
    
    % set up the split end
    split_end = split_start + progressStepSize - 1;
    if split_end > numOfVoxels
        split_end = numOfVoxels;
    end
    
    % fit the split
    parfor i=split_start:split_end
        
        % get the MR signals for the voxel i
        voxel = roi(i,:)';
        
        % fit the voxel
        if model.noOfStages == 2
            [gsps(i,:), fobj_gs(i), mlps(i,:), fobj_ml(i), error_code(i)] = ThreeStageFittingVoxel(voxel, protocol, model);
        else
            [gsps(i,:), fobj_gs(i), mlps(i,:), fobj_ml(i), error_code(i), mcmcps(i,:,:)] = ThreeStageFittingVoxel(voxel, protocol, model);
        end
        
%         % report to the progress monitor
%         if mod(i, progressStepSize)==0
%             ppm.increment();
%         end
        
    end  % end parfor
    
    % save the temporary results of the split
    if model.noOfStages == 2
        save(outputfile, 'split_end', 'model', 'gsps', 'fobj_gs', 'mlps', 'fobj_ml', 'error_code');
    else
        save(outputfile, 'split_end', 'model', 'gsps', 'fobj_gs', 'mlps', 'fobj_ml', 'mcmcps', 'error_code');
    end
    
end

toc

% ppm.delete();

% save the fitted parameters
if model.noOfStages == 2
    save(outputfile, 'model', 'gsps', 'fobj_gs', 'mlps', 'fobj_ml', 'roi', 'bval', 'bvec', ...
        'error_code', 'sz_data', 'mask_locs');
else
    save(outputfile, 'model', 'gsps', 'fobj_gs', 'mlps', 'fobj_ml', 'roi', 'bval', 'bvec', ...
        'mcmcps', 'error_code', 'sz_data', 'mask_locs');
end

% close the parallel pool
matlabpool close