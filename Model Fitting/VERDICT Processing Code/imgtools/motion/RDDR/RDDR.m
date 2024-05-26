function [I_reg, globalres] = RDDR(Images,Main)
% RDDR  Robust Data Decomposition Registration
%   [I_reg, globalres] = RDDR(Images, Main)
% 
% Images [ny nx nframe]  images to registered
% Main structure (see below for fields), additional field for results
% directory
%  Main.ResDir - if absent will call uigetdir
% 
% Example
% =======
%  dinfo = datparse ;
%  [vol, mat, locs] = d2mat(dinfo,{'dyn'},'resize',[128 128],'op',fp') ;
%  [Ireg] = RDDR(vol) ;
%  writeDicom(Ireg, dinfo, locs, 'positive')
%
% Adapted by David Atkinson from RDDReg written by Valentin Hamy.
% Also uses PROPACK and Residual Complexity code (see below).
%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                         %
%     Robust Data Decomposition Registration (RDDR) for breathhold        % 
%                 DCE-MR time series registration                         %
%               Valentin Hamy, UCL CMI - 08/09/2011                       %
%                                                                         %
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%
% 
% Method similar to progressive principal component registration (PPCR)
% involving low rank/spares matrix decomposition (LRSMD). The result from
% decomposition is then registered to a single time point and the
% corresponding deformation fields are then applied to the original input
% data.
%
% Registration achieved using minimization of residual complexity
% (cf. https://sites.google.com/site/myronenko/home)
%
%
% MAIN: Regsitration Parameters along with display options can be specified
% in MAIN structure (see code for field names)
%
% Optional Parameter [default value]
% MAIN.SIMILARITY: similarity measure (e.g. SSD, CC, SAD, [RC], CD2, MS,
% MI) - !!! The use of NMI isn't currently available (will require CMIC
% NiftyReg to be installed) !!!
% MAIN.ALPHA         : regularisation parameter in RC minimization [0.05]
% MAIN.SUBDIVIDE     : number of resoltuion levels [2]
% MAIN.OKNO: mesh window size, the smaller it is the more complex
% deformations are possible [4]
% MAIN.BE            : transformation regularization weight on bending energy, 0 for none [0.05]
% MAIN.SINGLE        : show mesh transformation at every iteration of RC
% minimization [0]
% MAIN.NITER         : Number of iterations in RDDR [10]
% MAIN.MAXLAMBDA     : Modifies multiplicative factor for the maximum value of
% RPCA trade-off parameter, lambda [12.5]
% MAIN.RANKTHRESHOLD : Modifies Threshold on LR component rank when
% initiating progress (additional tuning of information appearing in LR)
% MAIN.DISPLAY       : 1 show RPCA decomposition video (and Optical Flow
% registration process if run); 2 also saves result half way through
% registration [0] 
% MAIN.PROGRESS      : show cost function value evolution, etc [0]
% MAIN.RESAMPLE      : downsample images if bigger than 256*256 [0]
% MAIN.REG : choose registration approach (accumulated memory, all to
% first...) [0]
% MAIN.GROUP         :  Set up registration (0,1, or 2) a method of groupwise
% registration: 0 - All to the median, 1 - All to first, 2- next to the
% previous frame, 3 - accumalated memory (MAIN.GROUP only used if main.reg = 1) [0]  
% -- Warning : For cases 2 and 3 no parallel processing implemented yet --
% MAIN.FILTERBLUR    : filters out frames blured by intra-frame motion, likely
% to mislead registration (not complete) [0] 
% MAIN.TRANSLATION   : when = 1 turns on preliminary rigid registration [0]
% MAIN.OPTIC_FLOW    : offers the possibility to run generalized optical flow
% registration at the end of the registration loop (not complete) [0]
% MAIN.OUTPUTDEF     : Saves deformation field if on
% 
% WARNING: Code Tested on Windows platform. Modifications might be
% necessary on other platforms (mex files in particular...)
%
% 13-02-11 Subtract mean from image before applaying decomposition
%          and account for the number of timepoints in the value of Lambda
% 13-02-12 Test with preprocessing accounting for translation...
%
% 16-02-12 Rank checking (for low rank component) at the first iteration if
%          r<Nt/4 repeat the decomposition process with increased lambda
%
% 30-05-12 Incorporation of Optical Flow Registration (cf. Odille et al. 
%          Assessment of small bowel motility, MRM 2011)
%
% 08-06-12 Update: input data parsing
%
% 13-08-12 Update: Logarithmic distribution of Lambda value => reduced step
%          between higher value of lambda
%
% 12-11-12 Update: Use of dicom field RescaleSlope for intensities adjustment in Philips data
%
% 05-12-12 Update: addition of 3rd option in display -> save result after 6
%          iteration to leave potential motility bits preserved if
%          registering dynamic small bowel sequence
%
% 06-12-12 Update: notification of measure of sparseness (% of nnz) for S component
%          after every decomposition step.
%
% 18/12/12 Update: can save deformation field if needed
%
% 04/12/12 Update; flexibility on RankThreshold when initiating process





% Set up registration parameters
global main optim

if nargin >= 2
    main = Main;
    if ~isfield(Main,'similarity'); main.similarity = 'rc'; end
    if ~isfield(Main,'alpha'); main.alpha = 0.05; end
    if ~isfield(Main,'subdivide'); main.subdivide = 2; end
    if ~isfield(Main,'okno'); main.okno = 4; end
    if ~isfield(Main,'be'); main.be = 0.05; end
    if ~isfield(Main,'single'); main.single = 0; end
    if ~isfield(Main,'Niter'); main.Niter = 10; end
    if ~isfield(Main,'MaxLambda'); main.MaxLambda = 12.5; end
    if ~isfield(Main,'RankThreshold'); main.RankThreshold = -1; end
    if ~isfield(Main,'display'); main.display = 1; end
    if ~isfield(Main,'progress'); main.progress = 0; end
    if ~isfield(Main,'Resample'); main.Resample = 0; end
    if ~isfield(Main,'reg'); main.reg = 0; end
    if ~isfield(Main,'group'); main.group = 0; end
    if ~isfield(Main,'filterBlur'); main.filterBlur = 0; end
    if ~isfield(Main,'translation'); main.translation = 0; end
    if ~isfield(Main,'optic_flow'); main.optic_flow = 0; end
    if ~isfield(Main,'outputDef'); main.outputDef = 0; end
    if ~isfield(Main,'ResDir'); main.ResDir = '' ; end
else
    
    % Main settings
    main.similarity = 'rc'; % similarity measure, e.g. SSD, RC, MS, NMI
    main.alpha = 0.05;      % regularisation parameter
    main.subdivide = 2;     % use 2 hierarchical levels
    main.okno = 4;          % mesh window size, the smaller it is the more complex deformations are possible
    main.be = 0.05;     % transformation regularization weight, 0 for none
    main.single = 0;        % show mesh transformation at every iteration of RC minimization
    main.Niter = 10;        % number of iteration inRDDR (10 works well in gen but may try others values dep on application)
    main.MaxLambda = 12.5;
    main.RankThreshold = -1;% value of threshold on LR component rank when starting process (if = -1 use default threshold [Nt/4]);
    main.display = 1;       % show RPCA decomposition video (and Optical Flow registration process if run)
    main.progress = 0;      % show cost function value evolution, etc
    main.Resample = 0;      % resample images if bigger than 256*256
    main.reg = 0;           % choose registration approach (accumulated memory, all to first...)
    main.group = 0;         % Set up registration (0,1, or 2) a method of groupwise registration (main.group only used if main.reg = 1):
    % 0 - all to the closest to median
    % 1 - all to first
    % 2 - the next to the previous frame
    % 3 - the next to the moving average of the previous frames (accumulated memory)
    main.filterBlur = 0;    % filters out frames blured by intra-frame motion, likely to mislead registration (not accounting for the 1st 10 time-points where relatively high tmp resolution is needed)
    main.translation = 0;   % runs preliminary rigid registration (or not)
    main.optic_flow = 0;    % offers the possibility to run generalized optical flow registration at the end of the registration loop
    main.outputDef = 0;     % Saves Deformation Field if on
    main.ResDir = '' ; 
end

% Optimization settings
optim.maxsteps = 500;     % maximum number of iterations at each hierarchical level
optim.fundif = 1e-5;      % tolerance (stopping criterion)
optim.gamma = 1;          % initial optimization step size 
optim.anneal = 0.1;       % annealing rate on the optimization step 

if nargin > 2
    error('Expected 1 or 2 input arguments');
end

% Output directory
ResDir = main.ResDir ;
if ~exist(ResDir,'dir')
    if ispref('RDDR','ResDir')
        ResDir = getpref('RDDR','ResDir') ;
    end
    ResDir = uigetdir(ResDir,'Select folder for outputs') ;
    if ResDir == 0
        return
    else
        setpref('RDDR','ResDir',ResDir)
        main.ResDir = ResDir ;
    end
end

% Prior operation for use of //-processing
if matlabpool('size') == 0
    matlabpool;
end


[Ny Nx Nt] = size(Images) ;

tic; % Initialize timer
globalres = {}; % Initialize deformation field
% Niter = 10; % ceil(Nt/10); % Number of iteration
% if Niter > 10; Niter = 10; end
% if Niter < 5; Niter = 5; end

% Remove images with intra-frame motion (gasps in multiple breath-holds)
% if main.filterBlur
%     [idx,Images] = find_resp(Images,3,main.display);
%     DicomInfos(idx)=[]; DicomList(idx)=[];
%     fprintf('Filtered out %i images with intra-frame motion (indexes: ',Nt - size(Images,3));
%     for del_frame = 1:length(idx); fprintf('%i ', idx(del_frame)); end; fprintf(')\n');
%     Nt = size(Images,3);
%     if main.display
%         figure, montage(reshape(mat2gray(Images),Nx,Ny,1,Nt));
%     end
% end
   
% Clear/create result directory
if ~isdir([ResDir,'result\RDDReg\RDDReg',num2str(Nt),'\'])
  mkdir([ResDir,'result\RDDReg\RDDReg',num2str(Nt),'\']);
else
    if ~isempty(ls([ResDir,'result\RDDReg\RDDReg',num2str(Nt),'/*.dcm']))
        keydown = input(['Files already present in result folder, ', ...
        'are you sure you want to delete these and restart registration? (y/[n]): '], 's');
        if strcmp(keydown,'y') || strcmp(keydown,'Y') || isempty(keydown)
            delete([ResDir,'result\RDDReg\RDDReg',num2str(Nt),'\*']);
            if exist([ResDir,'result\RDDReg\RDDReg',num2str(Nt),'_lessIteration2preserveMotility\'],'dir')
                delete([ResDir,'result\RDDReg\RDDReg',num2str(Nt),'_lessIteration2preserveMotility\*']);
                rmdir([ResDir,'result\RDDReg\RDDReg',num2str(Nt),'_lessIteration2preserveMotility'],'s');
            end
        else
            error('as you wish...');
        end
    end
end

% Parameter for data decompositon
L = 1/sqrt(max(Nx*Ny,Nt));
% Lambda = linspace(L,2*L,Niter);
Lambda = L*log(linspace(exp(1),main.MaxLambda,main.Niter));
if main.RankThreshold == -1
    RankThreshold = Nt/4;
else
    RankThreshold = main.RankThreshold;
end

% Registration loop
if main.translation
    I_reg = CompensateTranslation(Images);
else
    I_reg = Images;
end
fprintf('Sarting Registration with %i iteration(s)\n', main.Niter);
fprintf('Parameters: okno %i; subdivide %i; lambda %i; single %i; progress %i; \n', ...
    main.okno,main.subdivide,main.be,main.single,main.progress);
fprintf('            reg %i; group %i; outputDef %i\n', ...
    main.reg, main.group, main.outputDef);
for iter = 1:main.Niter
    fprintf('-- Iteration: %i --\n', iter);
    
    % % Robust PCA
    lambda = Lambda(iter);
    
    I_timeSlice = I_reg;
    I_timeSlice = mat2gray(I_timeSlice);
    
    if iter == 1
        [LR_mat,S_mat] = RobustPCA(I_timeSlice,lambda,[],1e-7,200,0);
        rk = rank(LR_mat);
        disp(['Rank of LR component ',num2str(rk)]);
        % Use a criterion on rank to initiate the process (may need to find a more accurate approach in the future)
        while rk < RankThreshold || rk > 1.2*RankThreshold
            % Test for application to motility - only case where lambda had to be decreased so far...
            if rk < RankThreshold
                lambda = 1.05*lambda;
            elseif rk > 1.2*RankThreshold
                lambda = 0.9*lambda;
            end
            [LR_mat,S_mat] = RobustPCA(I_timeSlice,lambda,[],1e-7,200,0);
            rk = rank(LR_mat);
            disp(['Rank of LR component ',num2str(rk)]);
            Lambda = lambda*log(linspace(exp(1),main.MaxLambda,main.Niter));
            
        end
        if main.display >= 1
            map = getmap;
            mov = immovie(uint8(reshape(gray2ind(I_timeSlice,150),Nx,Ny,1,Nt)),map);
            movie2avi(mov, [ResDir,'result\RDDReg\Data_',number2string(iter),'_',num2str(floor(lambda*1000)),'.avi'], 'compression', 'None');
            mov = immovie(uint8(reshape(gray2ind(LR_mat,150),Nx,Ny,1,Nt)),map);
            movie2avi(mov, [ResDir,'result\RDDReg\LR_',number2string(iter),'_',num2str(floor(lambda*1000)),'.avi'], 'compression', 'None');
            mov = immovie(uint8(reshape(gray2ind(S_mat,150),Nx,Ny,1,Nt)),map);
            movie2avi(mov, [ResDir,'result\RDDReg\S_',number2string(iter),'_',num2str(floor(lambda*1000)),'.avi'], 'compression', 'None');
        end            
    else
        if main.display >= 1
            map = getmap;
            [LR_mat,S_mat] = RobustPCA(I_timeSlice,lambda,[],1e-7,200,0);
            mov = immovie(uint8(reshape(gray2ind(I_timeSlice,150),Nx,Ny,1,Nt)),map);
            movie2avi(mov, [ResDir,'result\RDDReg\Data_',number2string(iter),'_',num2str(floor(lambda*1000)),'.avi'], 'compression', 'None');
            mov = immovie(uint8(reshape(gray2ind(LR_mat,150),Nx,Ny,1,Nt)),map);
            movie2avi(mov, [ResDir,'result\RDDReg\LR_',number2string(iter),'_',num2str(floor(lambda*1000)),'.avi'], 'compression', 'None');
            mov = immovie(uint8(reshape(gray2ind(S_mat,150),Nx,Ny,1,Nt)),map);
            movie2avi(mov, [ResDir,'result\RDDReg\S_',number2string(iter),'_',num2str(floor(lambda*1000)),'.avi'], 'compression', 'None');
        else    
            [LR_mat,S_mat] = RobustPCA(I_timeSlice,lambda,[],1e-7,200,0); 
        end
        rk = rank(LR_mat);
        disp(['Rank of LR component ',num2str(rk)]);
    end
    disp(['Sparseness of S component ',num2str(100*nnz(S_mat)/(Nx*Ny*Nt))]);
    
    % Save Result into anchor data
    I_anchor = reshape(LR_mat,[Nx,Ny,Nt]);
    
    % % Compute deformation field using residual complexity (or Optical Flow Registration)
    % if main.optic_flow
    %     [~,I_reg] = OF_registrationcommand(DicomInfos,I_reg);
    %     % Find a way to combine optical flow deformation field with
    %     % globalres...
    % end
    [res,~] = registrationcommand(I_anchor);
    [globalres,I_reg] = transformationcommand(res,globalres,Images);
    
    if main.display >= 1
        map = getmap;
        [~,tmp] = transformationcommand(globalres,[],I_anchor);
        mov = immovie(uint8(reshape(gray2ind(tmp,150),Nx,Ny,1,Nt)),map);
        movie2avi(mov, [ResDir,'result\RDDReg\tmp_',number2string(iter),'_',num2str(floor(lambda*1000)),'.avi'], 'compression', 'None'); 
    end
    
    % Visual Assessment
    if  main.display == 2
        fprintf('Saving Temporary Results...\n')
        mkdir([ResDir,'result\RDDReg\RDDReg',num2str(Nt),'_Iter',num2str(iter)]);
        for i = 1:Nt
            [~,name,~] = fileparts(DicomList(i).name);
            dicomwrite(uint16(I_reg(:,:,i)),[ResDir,'result\RDDReg\RDDReg',num2str(Nt),'_Iter',num2str(iter),'\',name,'_Iter',num2str(iter),'.dcm'], ...
                DicomInfos{i}, 'CreateMode', 'copy');
        end
    end
    if (100*nnz(S_mat))/(Nx*Ny*Nt) < 15 && main.display == 3
        fprintf('Saving Temporary Results...\n')
        mkdir([ResDir,'result\RDDReg\RDDReg',num2str(Nt),'_lessIteration2preserveMotility_Iter',num2str(iter),'\']);
        for i = 1:Nt
            [~,name,~] = fileparts(DicomList(i).name);
            dicomwrite(uint16(I_reg(:,:,i)),[ResDir,'result\RDDReg\RDDReg',num2str(Nt),'_lessIteration2preserveMotility_Iter',num2str(iter),'\',name,'_Iter',num2str(iter),'.dcm'], ...
                DicomInfos{i}, 'CreateMode', 'copy');
            if main.outputDef
                save([ResDir,'result\RDDReg\RDDReg',num2str(Nt),'_lessIteration2preserveMotility_Iter',num2str(iter),'\DefField_RDDReg.mat'],'globalres','main');
            end
        end
        break;
    end
    % Early stop if too few information in sparse component
    if (100*nnz(S_mat))/(Nx*Ny*Nt) < 10 && main.display == 3
        break;
    end
end

% Close workers for parallel processing
if matlabpool('size') > 0
    matlabpool close;
end

% % Final regsitration step (optional)
% if main.optic_flow
%     [~,I_reg] = OF_registrationcommand(DicomInfos,I_reg);
%     % Find a way to combine optical flow deformation field with
%     % globalres...
% end

elapsed_time = toc;
elapsed_time_min = floor(elapsed_time/60);
elapsed_time_sec = elapsed_time - elapsed_time_min*60;
fprintf('Total time: %d min %2.1f sec\n', ...
    elapsed_time_min, elapsed_time_sec );


if main.outputDef
    fprintf('Saving Deformation Field...\n')

    SavedData.Descrition = ['Result from RDDR Applied to time series in',ResDir];
    SavedData.PatientName = DicomInfos.PatientName;
    SavedData.SeriesName = DicomInfos.SeriesName;
    if main.reg == 1 && (main.group == 0 || main.group == 1)
        SavedData.Ref = main.Ref; %do it directly in main...
    end
    SavedData.main = main;
    SavedData.optim = optim;
    SavedData.okno = globalres.okno;
    SavedData.DefX = globalres.X;
    SavedData.DefY = globalres.Y;
    SavedData.Elapsed_Time = [elapsed_time_min,elapsed_time_sec];
    save([ResDir,'result\RDDReg\RDDReg',num2str(Nt),'\SavedOutput_RDDReg.mat'],'SavedData');
end

% %% Nested Functions %%%

% Preprocessing compensating for potential rigid displacement
function I_comp = CompensateTranslation(Source)
I_comp = mirt2D_translation2(Source);


% Apply registration using residual complexity
function [res,ImagesReg] = registrationcommand(Images)
global main optim
padsize = main.okno;
I_src = padarray(Images,[padsize padsize 0]);
if ~strcmp(main.similarity,'nmi') && ~strcmp(main.similarity,'NMI')
    if main.reg == 0
        res = mirt2Dgroup_sequence(I_src,main,optim);
    elseif main.reg == 1
        res = mirt2Dgroup_frame(I_src,main,optim);
    end
else % use nifti_reg if nmi chosen
    warning(['Nifti_reg implementation incomplete'])
end



if ~strcmp(main.similarity,'nmi') && ~strcmp(main.similarity,'NMI')
    [~,ImagesReg] = mirt2Dgroup_transform(I_src, res, main.subdivide);
else
    [~,ImagesReg] = mirt2Dgroup_transform(I_src, res, main.subdivide);
end
ImagesReg(isnan(ImagesReg))= 0;
ImagesReg = reshape(ImagesReg(padsize+1:end-padsize,padsize+1:end-padsize,:),size(Images));

 
 
% Update and apply transformation
function [globalres,I_reg] = transformationcommand(res,globalres,I)
global main
padsize = main.okno;

% % Test to check there's indeed a difference between res and global res
% [~,I_reg_tmp] = mirt2Dgroup_transform(padarray(I,[10 10 0]), res, main.subdivide);
% I_reg_tmp(isnan(I_reg_tmp))= 0;
% I_reg_test = I_reg_tmp(11:end-10,11:end-10,:);

if isempty(globalres)
    if ~strcmp(main.similarity,'nmi') && ~strcmp(main.similarity,'NMI')
        [globalres,I_reg_tmp] = mirt2Dgroup_transform(padarray(I,[padsize padsize 0]), res, main.subdivide);
    else
        [globalres,I_reg_tmp] = mirt2Dgroup_transform(padarray(I,[padsize padsize 0]), res, main.subdivide);
    end
    I_reg_tmp(isnan(I_reg_tmp))= 0;
    I_reg = I_reg_tmp(padsize+1:end-padsize,padsize+1:end-padsize,:);
else
    globalres = cat(2,globalres,res);
    if ~strcmp(main.similarity,'nmi') && ~strcmp(main.similarity,'NMI')
        [globalres,I_reg_tmp] = mirt2Dgroup_transform(padarray(I,[padsize padsize 0]),globalres,main.subdivide);
    else
        [globalres,I_reg_tmp] = mirt2Dgroup_transform(padarray(I,[padsize padsize 0]),globalres,main.subdivide);
    end
    I_reg_tmp(isnan(I_reg_tmp))= 0;
    I_reg = I_reg_tmp(padsize+1:end-padsize,padsize+1:end-padsize,:);
end

%  figure, montage(reshape(I_reg-I_reg_test,size(I,1),size(I,2),1,size(I,3)),'DisplayRange',[]); drawnow;
 
% Apply registration using generalized optical flow
function [res,ImagesReg] = OF_registrationcommand(DicomInfos,Images)
global main

% Spatial/temporal resolution
dt = double(DicomInfos{1}.RepetitionTime * DicomInfos{1}.AcquisitionMatrix(3) ) / 1000;
dx = DicomInfos{1}.PixelSpacing(1) * 2;

% Find image closest to the median image
Imedian = median(Images,3);
dist    = zeros(1,size(Images,3));
for t=1:size(Images,3)
    It = Images(:,:,t);
    dist(t) = norm(It(:) - Imedian(:));
end
[~,ref] = min(dist); % the reference image for registration

% Choose kernel size for image interpolations
W = 6; % windowed sinc (lanczos window) interpolation
nbins   = 2^16;
xList = -W/2 : W/(nbins-1) : W/2 ;
if(W==2)
    LookupTable = 1 - abs(xList);
else
    LookupTable = sinc(xList) .* sinc( xList / (W/2) ); % lanczos-sinc
end

Interpolator.LookupTable = LookupTable;
Interpolator.nbins       = nbins;
Interpolator.W           = W;

% Registration parameters
Param.lambdaU   = 1.e-4; % smoothness weight of the displacement fields
Param.lambdaC   = 5.e-6; % smoothness weight of the map of intensity changes
Param.display   = main.display;
Param.dx        = dx; % spatial resolution in mm/pixel
Param.dt        = dt; % temporal resolution in s/pixel
Param.V0        = Inf; % mm/s
Param.ref       = ref;
Param.ResolutionLevels	= [1/8 1/4 1/2 1];
Param.NbLoops           = 4;
Param.Interpolator      = Interpolator;
Param.RegularizerOrder  = 2; % 1: minimize norm of the 1st order derivative of displacements and intensity maps
                             % 2: minimize norm of the 2nd order derivative of displacements and intensity maps

 % Run registration
[Ux,Uy,~,ImagesReg] = generalized_optical_flow_2D(Images, Param);
res.Ux = Ux;
res.Uy = Uy;


% Simply defines map for videoa
function map = getmap
    map = [
             0         0         0;
             0.0078    0.0078    0.0078;
             0.0196    0.0196    0.0196;
             0.0314    0.0314    0.0314;
             0.0431    0.0431    0.0431;
             0.0549    0.0549    0.0549;
             0.0667    0.0667    0.0667;
             0.0784    0.0784    0.0784;
             0.0902    0.0902    0.0902;
             0.1020    0.1020    0.1020;
             0.1137    0.1137    0.1137;
             0.1255    0.1255    0.1255;
             0.1333    0.1333    0.1333;
             0.1451    0.1451    0.1451;
             0.1569    0.1569    0.1569;
             0.1686    0.1686    0.1686;
             0.1804    0.1804    0.1804;
             0.1922    0.1922    0.1922;
             0.2039    0.2039    0.2039;
             0.2157    0.2157    0.2157;
             0.2275    0.2275    0.2275;
             0.2392    0.2392    0.2392;
             0.2510    0.2510    0.2510;
             0.2588    0.2588    0.2588;
             0.2706    0.2706    0.2706;
             0.2824    0.2824    0.2824;
             0.2941    0.2941    0.2941;
             0.3059    0.3059    0.3059;
             0.3176    0.3176    0.3176;
             0.3294    0.3294    0.3294;
             0.3412    0.3412    0.3412;
             0.3529    0.3529    0.3529;
             0.3647    0.3647    0.3647;
             0.3765    0.3765    0.3765;
             0.3843    0.3843    0.3843;
             0.3961    0.3961    0.3961;
             0.4078    0.4078    0.4078;
             0.4196    0.4196    0.4196;
             0.4314    0.4314    0.4314;
             0.4431    0.4431    0.4431;
             0.4549    0.4549    0.4549;
             0.4667    0.4667    0.4667;
             0.4784    0.4784    0.4784;
             0.4902    0.4902    0.4902;
             0.5020    0.5020    0.5020;
             0.5098    0.5098    0.5098;
             0.5216    0.5216    0.5216;
             0.5333    0.5333    0.5333;
             0.5451    0.5451    0.5451;
             0.5569    0.5569    0.5569;
             0.5686    0.5686    0.5686;
             0.5804    0.5804    0.5804;
             0.5922    0.5922    0.5922;
             0.6039    0.6039    0.6039;
             0.6157    0.6157    0.6157;
             0.6275    0.6275    0.6275;
             0.6353    0.6353    0.6353;
             0.6471    0.6471    0.6471;
             0.6588    0.6588    0.6588;
             0.6706    0.6706    0.6706;
             0.6824    0.6824    0.6824;
             0.6941    0.6941    0.6941;
             0.7059    0.7059    0.7059;
             0.7176    0.7176    0.7176;
             0.7294    0.7294    0.7294;
             0.7412    0.7412    0.7412;
             0.7529    0.7529    0.7529;
             0.7608    0.7608    0.7608;
             0.7725    0.7725    0.7725;
             0.7843    0.7843    0.7843;
             0.7961    0.7961    0.7961;
             0.8078    0.8078    0.8078;
             0.8196    0.8196    0.8196;
             0.8314    0.8314    0.8314;
             0.8431    0.8431    0.8431;
             0.8549    0.8549    0.8549;
             0.8667    0.8667    0.8667;
             0.8784    0.8784    0.8784;
             0.8863    0.8863    0.8863;
             0.8980    0.8980    0.8980;
             0.9098    0.9098    0.9098;
             0.9216    0.9216    0.9216;
             0.9333    0.9333    0.9333;
             0.9451    0.9451    0.9451;
             0.9569    0.9569    0.9569;
             0.9686    0.9686    0.9686;
             0.9804    0.9804    0.9804;
             0.9922    0.9922    0.9922;
             1.0000    1.0000    1.0000];
