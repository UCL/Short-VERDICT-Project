function [scheme, Y,fIC, fEES, fVASC, R, rmse] = verdict_process_Adam(dfolder, output_folder, opts)
% VERDICT_PROCESS VERDICT processing 
%
% See wrapping function VERDICT for how to call this function
%
% Processing of VERDICT MRI data. Fits a ball, sphere and astrosticks
% tissue model to diffusion data from 5 scans. Outputs include fIC map.
% The acquisition scheme is currently hard-coded for the scanners listed in
% bv2scheme and any scheme file present is ignored. Results may differ 
% if the assumed scheme was not used for the actual measurement.
%
% verdict_process(dfolder, output_folder, Name, Value, ...)
%
% Loads data from dfolder (Enhanced or Classic DICOM files in any order).
% Other DICOMS may be present but there should be exactly one dataset for
% each of the 5 VERDICT scans (b=90, 500, 1500, 2000, 3000 each with a
% b=0).
% Optional registration (translations only, based on central region of
% central slice, b=0 to b=0 alignment with transformation applied to DW data).
% Fit of model to data. Model uses scheme (b-value, G, deltas)
% Default solver is lsqnonneg with Tikonhov regularisation. 
% Output of maps.
% Colormaps are experimental and not tested.
%
% Uses DICOM ManufacturerModelName to choose a scheme unless forceXNATscheme 
% is true. ("XNAT" here refers to reference processing used for Singh et al
% Radiology 2022)
%
% Outputs a PDF report. Logging previously used the Advanced Logger 
% for MATLAB  (available on FileExchange / GitHub) but removed to aid with
% deployment to Docker
%
%
% When called from the command line (e.g. in Docker), Name and Value must
% be character strings
%
% The divergent colourmap comes from colorcet:
% Reference:
% Peter Kovesi. Good Colour Maps: How to Design Them.
% arXiv:1509.03700 [cs.GR] 2015
% https://arxiv.org/abs/1509.03700
%
% Examples
%  % This Example will add DICOMs and other output to the folder dfolder
%  dfolder = '/User/myname/data/STUDYID-005' ;
%  verdict_process(dfolder, dfolder, outputDicom=true)
%
%  verdict_process(dfolder, tempdir, solver='SPAMS')    % MATLAB mode only
%  verdict_process(dfolder, tempdir, 'solver', 'SPAMS') 
%
% As a command line:
% verdict_process 'inputfolder' 'outputfolder' solver SPAMS  addb0ToScheme true
%
% Configurations
% --------------
% XNAT results closest (swapping in volumes from XNAT registrations)
% swapinvXNAT=true, register=false, usedirecdiff=true, solver='SPAMS', forceXNATscheme=true, addb0ToScheme=true 
% XNATmimic = {'register','false','swapinvXNAT','true','usedirecdiff','true','solver','SPAMS','forceXNATscheme','true','addb0ToScheme','true'}
%
% As above, but correct scanner (scheme) with b=0 added.
% swapinvXNAT=true; register=false; usedirecdiff=true; solver='SPAMS'; forceXNATscheme=false; addb0ToScheme=true ;
%
% XNAT registered input, correct scheme (with added b=0), lsqnoninTokonohv reg
% swapinvXNAT=true; register=false; usedirecdiff=true; solver='lsqnonnegTikonhov'; forceXNATscheme=false; addb0ToScheme=true ;
%
% MATLAB process (the default)
% swapinvXNAT=false, register=true, usedirecdiff=false, solver='lsqnonnegTikonhov', forceXNATscheme=false, addb0ToScheme=true 
%
% Swapping in a file vMAT.mat. Should be a file vMAT.mat containing the
% variable vMAT such that vb0 = vMAT(:,:,:,:,1)  and vbv = vMAT(:,:,:,:,2)
% corresponding to ascending b-values 90,500,1500,2000,3000
% swapinvMAT=true, usedirecdiff=false
%
%
% David Atkinson
%
% See also bv2scheme verdict_fit  sviewer  getSeriesVERDICT verdict


%% David Code

% input arguments will be characters when called from command line in
% deployed mode. Converted below to boolean or numerical if necessary

% The arguments block was moved from here to the wrapper function verdict to 
% allow verdict.m and pcode for verdict_process

opts = convertCharsToLogical(opts) ; 
if ischar(opts.maskhwmm)
    opts.maskhwmm = str2double(opts.maskhwmm) ;
end
if isfield(opts,'vBaseSeriesNumber') && ischar(opts.vBaseSeriesNumber)
    opts.vBaseSeriesNumber = str2double(opts.vBaseSeriesNumber) ;
end
if isfield(opts,'fICReconNumber') && ischar(opts.fICReconNumber)
    opts.fICReconNumber = str2double(opts.fICReconNumber) ;
end
if isfield(opts,'vADCbmax') && ischar(opts.vADCbmax)
    opts.vADCbmax = str2double(opts.vADCbmax) ;
end
if isfield(opts,'allowedSeriesNumbers') && ischar(opts.allowedSeriesNumbers)
    opts.allowedSeriesNumbers = str2double(opts.allowedSeriesNumbers) ;
end
if isfield(opts,'ncompart') && ischar(opts.ncompart)
    opts.ncompart = str2double(opts.ncompart) ;
end

% 
% % Set timestamp tstr for output folder naming 
% tnow = datetime ;
% tstr_nice = char(tnow) ;
% tnow.Format='yyyy-MM-dd''T''HHmmss' ; % will form part of path name (no weird characters)
% tstr = char(tnow) ;
% 
% % Create subfolder for results (name using time string)
% if ~isfield(opts,'resultsFolderName') || isempty(opts.resultsFolderName)
%     opts.resultsFolderName = ['res-',tstr];
% end
% resultsFolder = fullfile(output_folder, opts.resultsFolderName) ;
% 
% 
% 
% verdictVersion = opts.verdictVersion ;
% % 
% % append(rpt, Paragraph(['This report was generated: ', tstr_nice]) )
% % append(rpt, Paragraph(' '))
% % append(rpt, Paragraph(['VERDICT version: ', verdictVersion ]))
% % append(rpt, Paragraph(' '))
% % append(rpt, Paragraph(['This report file name is: ', rptFFN]))
% % append(rpt, Paragraph(' '))
% % append(rpt, Paragraph(['The results folder is: ',resultsFolder]))

% % append(rpt,Chapter('Inputs'))

% if opts.quiet
%     figVisible = 'off' ; % figures do not display, but are still in report
% else
%     figVisible = 'on' ;
% end
% % 
% % 
% % if opts.swapinvXNAT
% %     if opts.register == true || opts.usedirecdiff == false
% %         msg = 'Cannot register or use iso diff when swapinvXNAT is true';
% %         warning(msg);
% %         append(rpt, Paragraph(msg)) ;
% %         append(rpt, Paragraph(' '))
% %     end
% % end
% % 
% % if opts.swapinvMAT
% %     if opts.swapinvXNAT
% %         msg = 'Cannot have both vMAT and vXNAT swap in.';
% %         warning(msg);
% %         append(rpt, Paragraph(msg)) ;
% %         append(rpt, Paragraph(' '))
% %     end
% %     if opts.usedirecdiff == true
% %         msg = 'vMAT does not have directional diffusion, cannot usedirecdiff.';
% %         warning(msg);
% %         append(rpt, Paragraph(msg)) ;
% %         append(rpt, Paragraph(' '))
% %     end
% % end


% 
% % VERDICT AMICO paper used 0.01:15.1 
% % In online AMICO linspace(0.01,20.1,20);
% % https://github.com/daducci/AMICO_matlab/blob/master/models/AMICO_VERDICTPROSTATE.m
% % Exact choice seems to make little difference to fIC.
% 
% rmseThreshold = 0.05 ; % Root Mean Square Error above this coloured rmseRGB
% rmseRGB = [0.8 0.8 0.8] ; % [0.8  0  0.8] is magenta-like
% 
% fw_pix = 1000 ; % width of larger figures in pixels
% 

switch opts.usedirecdiff
    case true
        nbd = 3 ; % diffusion weighted images per file
    case false
        nbd = 1 ; % isotropic diffusion weighted image
    otherwise
        warning('Unknown value of usedirecdiff')
end


if opts.addb0ToScheme
    naddb0 = 1 ; % number of b=0 added per series 
else
    naddb0 = 0 ;
end

% 
% if opts.forceXNATscheme == true
%     if opts.addb0ToScheme == false
%         warning('forceXNATscheme hence adding b=0 scans to scheme')
%     ends
%     naddb0 = 1 ; % number of b=0 added per series 
% end


% % append(rpt, Paragraph('Input Options'))
% % append(rpt, Paragraph(' '))
% % append(rpt, Paragraph(['Number of diffusion images used per Series (nbd): ',num2str(nbd)]))
% % append(rpt, Paragraph(['Number of b=0 added per Series: ',num2str(naddb0)]))
% % append(rpt, Paragraph(' '))

% % Report parameters in a table
% optsout = opts;
% optsout = rmfield(optsout,'allowedSeriesNumbers') ;
% pTObj = MATLABTable(struct2table(optsout, 'AsArray',true)) ;
% pTObj.Style = [pTObj.Style { FontSize('8pt')} ] ;
% append(rpt, pTObj)
% 
% append(rpt,Paragraph(' '))
% append(rpt,Paragraph(['Input data folder: ',dfolder]))
% append(rpt,Paragraph(' '))


% dicomdict("factory")




if opts.Rs
    Rs = opts.Rs;
else
    Rs = linspace(0.1,15.1,17) ; % radii used in fit.
end

%% CHECK IF RELEVANT DATA FOR PATIENT & SCHEME IS SAVED ALREADY

datafolder = [char(opts.parent_folder) '/MATLAB Saved VERDICT Data/' char(opts.schemename) '/' char(opts.PatientID  )];
loadsuccess = false;

if exist(datafolder, "dir")

    try
        % LOAD NECESSARY VARIABLES
        load([datafolder '/dinfo.mat'])
        load([datafolder '/geomout.mat'])
        load([datafolder '/Y.mat']);
        load([datafolder '/scheme.mat']);
        load([datafolder '/b0fromhighb.mat'])
        loadsuccess = true;
    catch
        loadsuccess = false;
    end

end

if ~loadsuccess

    %% GET VERDICT SERIES
    
    % I NEED TO WRITE SOME CODE TO GENERATE series_excludebvals FROM schemename
    % VARIABLE
    
    [dinfo, vSeriesNumbers, vBV, dinfoT2ax, T2axSeriesNumbers, vMAT, dscheme] = getSeriesVERDICT(dfolder, allowedSeriesNumbers = opts.allowedSeriesNumbers);
    
    
    % CODE HERE TO CHECK VERDICT SCHEME MATCHES expectation from schemename
    % INPUT

    % 1. Load saved scheme (from MLP tests folder)
    % 2. Add b values which don't match loaded scheme to exclude_bvals (compare to vBV)
    % 3. If scheme contains b value not in loaded scheme, throw error
    exclude_bvals = [];
    expected_scheme = load([char(opts.schemesfolder) '/' opts.schemename]).scheme;
    expected_bvals = [expected_scheme.bval];
    expected_bvals = expected_bvals(expected_bvals ~= 0);

    for bindx = 1:length(vBV(:,2))

        bval = vBV(bindx,2);
        
        if ~ismember(bval, expected_bvals)
            
            exclude_bvals = [exclude_bvals bval];
        
        end

    end


    for bindx =1:length(expected_bvals)

        bval = expected_bvals(bindx);

        if ~ismember(bval, vBV(:,2))
            error("Loaded data doesn't match scheme")
        end

    end
    
    if isempty(dinfo)
    % %     append(rpt,Paragraph(' '))
    % %     msg = 'NO INPUT FOUND. (Check file and path names / connection to networked or external drives)';
    % %     append(rpt,Paragraph(msg))
    
        disp('Exiting as no input found.')
        return
    end
    
    
    % % % Report exam details
    % % append(rpt,Paragraph(' '))
    % % append(rpt,Paragraph('DICOM details'))
    
    dfi = dicominfo(dinfo(1).Filename) ;
    % % if isfield(dfi,'PatientID')
    % %     append(rpt,Paragraph(['PatientID: ',dfi.PatientID]))
    % % end
    % % if isfield(dfi,'StudyDescription')
    % %     append(rpt,Paragraph(['StudyDescription: ',dfi.StudyDescription]))
    % % end
    % % if isfield(dfi,'InstitutionName')
    % %     append(rpt,Paragraph(['InstitutionName: ',dfi.InstitutionName]))
    % % end
    % % if isfield(dfi,'ProtocolName')
    % %     append(rpt,Paragraph(['ProtocolName: ',dfi.ProtocolName]))
    % % end
    % % if isfield(dfi,'Manufacturer')
    % %     append(rpt,Paragraph(['Manufacturer: ',dfi.Manufacturer]))
    % % end
    % % if isfield(dfi,'ManufacturerModelName')
    % %     append(rpt,Paragraph(['ManufacturerModelName: ',dfi.ManufacturerModelName]))
    % % end
    % % if isfield(dfi,'SoftwareVersions')
    % %     append(rpt,Paragraph(['SoftwareVersions: ',dfi.SoftwareVersions]))
    % % end
    % % append(rpt,Paragraph(' '))
    
    % % % Check dinfo for reasonable parameter values
    % % % Add to report and/or logging
    % % pcheck = checkV(dinfo) ;
    % % append(rpt, pcheck)
    
    
    [sortedBV, indsortedBV] = sort(vBV(:,2), opts.bvSortDirection) ;
    
    nSeries = length(vSeriesNumbers) ;
    
    iptprefsOrig = iptgetpref ;
    iptsetpref('ImshowBorder','tight')
    iptsetpref('ImshowInitialMagnification','fit') 
    
    if opts.register
        regTranslations = zeros([nSeries 2]) ;  % Registration transformations stored for report
        [optimizer,metric] = imregconfig('monomodal') ; % monomodal as b=0 to b=0
        % fw = fw_pix ;    % figure width in pixels
        % fh = fw/nSeries*3 ;
    % %     hfreg_check = figure(Name='Pre and Post Registration', Position=[250 500 fw round(fh)],Visible=figVisible);
    % %     treg = tiledlayout(3,nSeries,'TileSpacing','none','Padding','tight') ;
    % %     axrs = [] ;
    end
    
    % Pre-allocate variables used for checking data
    medianvb0 = zeros([1 nSeries]) ; medianvbv = zeros([1 nSeries]) ; plotbv = zeros([1 nSeries]) ;
    TE = zeros([1 nSeries]) ; TR = zeros([1 nSeries]) ;
    
    nDeltaFromDicom = 0 ;
    
    for iSeries = 1: nSeries
    
        sn_this = vSeriesNumbers(indsortedBV(iSeries)) ;
        bv_this = vBV(indsortedBV(iSeries),2) ;
    
        % Get b=0 data
        [vb0, mb0, b0loc] = d2mat(dinfo,{'slice','bv','series'},'bv',0, ...
            'series',sn_this,'op','fp') ;
    
   
        TR(indsortedBV(iSeries)) = dinfo(b0loc(1)).RepetitionTime ;
        if isfield(dinfo,'EffectiveEchoTime')
            TE(indsortedBV(iSeries)) = dinfo(b0loc(1)).EffectiveEchoTime ;
        end
    
        % Get b>0 data
        if opts.usedirecdiff
            [vbv, mbv, bvloc] = d2mat(dinfo,{'slice','bdirec','ddty','series'}, ...
                'ddty',1,'series',sn_this,'op','fp') ;
            if size(vbv,4)~=3, warning('Expected 3 diffusion directions'), end
        else
            [vbv, mbv, bvloc] = d2mat(dinfo,{'slice','ddty','series','bv'}, ...
                'series',sn_this, 'bv', bv_this, ...
                'ddty', 2, 'op','fp') ;
        end


        % 
        % if opts.swapinvXNAT
        %     % in the reference XNAT, b-value ordering is 3000 to 90
        %     switch bv_this
        %         case 90
        %             vb0 = vMAT(:,:,:,17) ;
        %             vbv = vMAT(:,:,:,18:20) ;
        %         case 500
        %             vb0 = vMAT(:,:,:,13) ;
        %             vbv = vMAT(:,:,:,14:16) ;
        %         case 1500
        %             vb0 = vMAT(:,:,:,9) ;
        %             vbv = vMAT(:,:,:,10:12) ;
        %         case 2000
        %             vb0 = vMAT(:,:,:,5) ;
        %             vbv = vMAT(:,:,:,6:8) ;
        %         case 3000
        %             vb0 = vMAT(:,:,:,1) ;
        %             vbv = vMAT(:,:,:,2:4) ;
        %         otherwise
        %             error(['Unrecognised bv_this: ',num2str(bv_this)])
        %     end
        % end
        % 
        % if opts.swapinvMAT
        %     % vMAT 
        %     switch bv_this
        %         case 90
        %             vb0 = vMAT(:,:,:,1,1) ;
        %             vbv = vMAT(:,:,:,1,2) ;
        %         case 500
        %             vb0 = vMAT(:,:,:,2,1) ;
        %             vbv = vMAT(:,:,:,2,2) ;
        %         case 1500
        %             vb0 = vMAT(:,:,:,3,1) ;
        %             vbv = vMAT(:,:,:,3,2) ;
        %         case 2000
        %             vb0 = vMAT(:,:,:,4,1) ;
        %             vbv = vMAT(:,:,:,4,2) ;
        %         case 3000
        %             vb0 = vMAT(:,:,:,5,1) ;
        %             vbv = vMAT(:,:,:,5,2) ;
        %         otherwise
        %             error(['Unrecognised bv_this: ',num2str(bv_this)])
        %     end
        % end
        % 
        % % compute medians to check signal
        % vexcend = vb0(:,:,3:end-2) ;
        % medianvb0(iSeries) = median(vexcend(:),'omitnan') ;
        % vexcend = vbv(:,:,3:end-2,:) ;
        % medianvbv(iSeries) = median(vexcend(:),'omitnan') ;
        % plotbv(iSeries) = bv_this ;

        %%
    
        if iSeries == 1 
    
            % SAVE b0 from highest b value 
            b0fromhighb = vb0;
    
            % first pass
            Y = zeros([size(vb0,[1 2 3]), (nbd + naddb0)*nSeries]) ;
    
            vb0tot = zeros(size(vb0));
            v2000 =  zeros(size(vb0)) ;
            v2000norm = zeros(size(v2000)) ;
    
            % these are output in a .mat file for testing/debugging
            vprereg  = zeros([size(vb0,[1 2 3]), (nbd + 1)*nSeries]) ;
            vpostreg = zeros([size(vb0,[1 2 3]), (nbd + 1)*nSeries]) ;
            
    
            % Find pixels in mask aroud image centre where prostate is assumed
            % to be located (within +/- maskhwmm) mm of centre.
            maskhw = floor(opts.maskhwmm/mb0.geom(1).PixelSpacing_HW(1)) ;
            [ny, nx, nz] = size(vb0,[1 2 3]) ;
    
            maskcentre = [ceil( (ny+1)/2 )  ceil((nx+1)/2) ] ;
    
            maskc = { max(1,maskcentre(1)-maskhw) : min(ny,maskcentre(1)+maskhw) , ...
                max(1,maskcentre(2)-maskhw) : min(nx,maskcentre(2)+maskhw) } ;
    
            reg_slice = ceil((nz+1)/2) ; % registration uses only one slice per series
    % %         append(rpt, Paragraph(['Registration slice number: ',num2str(reg_slice)])) ;
    
    
            vfixed = vb0(maskc{:},reg_slice) ; % fixed image for registration
            vfixedUpper = prctile(vfixed(:),98) ;
            
            dfullinf = dicominfo(dinfo(1).Filename) ;
            if isfield(dfullinf,'ManufacturerModelName')
                scanner = dfullinf.ManufacturerModelName ;
    % %             append(rpt, Paragraph(['scanner:', scanner])) ;
            else
                warning('Scanner (ManufacturerModelName) not known')
    % %             append(rpt, Paragraph('Scanner (ManufacturerModelName) not known'))
            end
    
            if opts.forceXNATscheme
    % %             append(rpt,Paragraph('! Forcing use of XNAT scheme file.'))
                scanner = 'XNAT' ;
            end
    
            if isfield(dfullinf,'BodyPartExamined')
                BodyPartExamined = dfullinf.BodyPartExamined ;
                if strcmp(BodyPartExamined,'KIDNEY' )
                    scanner = [scanner,'Renal'] ;
    % %                 append(rpt, Paragraph(['RENAL: scanner switched to:', scanner])) ;
                end
            else
                BodyPartExamined = 'UNKNOWN' ;
            end
    
            if ~isempty(opts.forcedSchemeName)
                scanner = opts.forcedSchemeName ;
    % %             append(rpt, Paragraph(['FORCED SCHEME: scanner set to:', scanner])) ;
            end
    
    
    
    % %         % Checks if a scheme file is present and reports only
    % %         % Note code still uses bv2scheme hard-coded values
    % %         [numFiles, fileNames] = findVerdictSchemeFiles(dfolder) ;
    % %         if numFiles == 1
    % %             [version, tableData] = readVerdictSchemeFile(fileNames{1}) ;
    % %             [meetsRequirements, messg] = checkVerdictSchemeTable(tableData, scanner) ;
    % %             if meetsRequirements
    % %                 mRstr = ['Scheme file meets requirements. Version: ',num2str(version)] ;
    % %                 append(rpt,Paragraph(mRstr))
    % %             else
    % %                 append(rpt,Paragraph(['Scheme file present but does not meet requirements: ', messg]))
    % %             end
    % %         elseif numFiles == 0
    % %             append(rpt,Paragraph('No scheme file present.'))
    % %         else
    % %             append(rpt,Paragraph('More than one scheme file was found.'))
    % %         end
    
            geomout = mb0.geom ; % used as reference and for writing DICOMs
            dinfoout = dinfo ; % for DICOM output
            locout = b0loc ;
            if ~isfield(opts,'vBaseSeriesNumber')
                if sn_this > 200
                    vbsn = floor(sn_this/100) ;
                else
                    vbsn = sn_this ;
                end
    
                opts.vBaseSeriesNumber = vbsn ;
            end
        end % iSeries == 1
    
    % %     % Check data is consistent and report
    % %     pcheck = checkV(geomout, mb0.geom) ;
    % %     append(rpt, pcheck)
    % %     pcheck = checkV(geomout, mbv.geom) ;
    % %     append(rpt, pcheck)
    % %     pcheck = checkV(vb0) ;
    % %     append(rpt,pcheck)
    % %     pcheck = checkV(vbv) ;
    % %     append(rpt,pcheck)
    % %     if ~strcmp(BodyPartExamined,'KIDNEY')
    % %         pcheck = checkV(mb0.geom,'axial') ;
    % %         append(rpt, pcheck)
    % %     end
       
    
        if opts.register
            vmoving = vb0(maskc{:},reg_slice) ;
            ylinePos = ceil(size(vmoving,1)/2) ;
            xlinePos = ceil(size(vmoving,2)/2) ;
            nxlim = size(vmoving,2) ;
            nylim = size(vmoving,1) ;
    
            % different b=0's do not always have the same scale - 
            % normalising first seems more robust for registration
    
            vmoving_toreg = mat2gray(vmoving,[0 double(prctile(vmoving(:),98))]) ;
            vfixed_toreg  = mat2gray(vfixed, [0 double(prctile(vfixed(:),98))]) ;
    
            tform = imregtform(vmoving_toreg,vfixed_toreg,'translation',optimizer,metric);
    
            regTranslations(iSeries,:) = tform.Translation ; % store for report
    
            vb0_reg = zeros(size(vb0)) ;
            vbv_reg = zeros(size(vbv)) ;
    
            for islice =1:size(vb0,3)
                vb0_reg(:,:,islice) = imwarp(vb0(:,:,islice),tform,"OutputView",imref2d(size(vb0(:,:,1)))) ;
                for ibd = 1:nbd
                    vbv_reg(:,:,islice,ibd) = imwarp(vbv(:,:,islice,ibd), tform,"OutputView",imref2d(size(vbv(:,:,1)))) ;
                end
            end
    
            % Figure to QA registration. Top row preserves scaling
            % Middle row is moving images
            % Bottom row is registered.
    % %         axr = nexttile(treg,tilenum(treg,1,iSeries)) ;
    % %         imshow(vmoving,[0 vfixedUpper],'Parent',axr) ;
    % %         axrs = [axrs axr];
    % % 
    % %         axr = nexttile(treg,tilenum(treg,2,iSeries)) ;
    % %         imshow(vmoving,[ ],'Parent',axr), hold on
    % %         plot(axr,[1 nxlim],[ylinePos ylinePos],'Color','y')
    % %         plot(axr,[xlinePos xlinePos],[1 nylim],'Color','y')
    % %         axrs = [axrs axr];
    % %         axr = nexttile(treg,tilenum(treg,3,iSeries)) ;
    % %         imshow(vb0_reg(maskc{:},reg_slice),[ ],'Parent',axr), hold on
    % %         plot(axr,[1 nxlim],[ylinePos ylinePos],'Color','y')
    % %         plot(axr,[xlinePos xlinePos],[1 nylim],'Color','y')
    % %         axrs = [axrs axr];
        else
            vb0_reg = vb0 ;
            vbv_reg = vbv ;
        end
    
        if abs(bv_this-2000) < 10
            v2000 = sum(vbv_reg,4) ;
            locb2000  = bvloc ;
            v2000norm = v2000 ./ vb0_reg ;
        end
    
        vb0tot = vb0tot + vb0_reg ;
    
        vnorm = vbv_reg./repmat(vb0_reg,[1 1 1 nbd]) ; % DW images, normalised by b=0
        bvinY_this = repmat(bv_this,[1 nbd]) ;
    
    
        if naddb0 == 1
            vnorm = cat(4,ones(size(vb0_reg)), vnorm ) ;
            bvinY_this = cat(2,0,bvinY_this) ;
        end
    
        bvinY(1, 1+(iSeries-1)*(nbd + naddb0) : iSeries*(nbd + naddb0)) = bvinY_this ;
        Y(:,:,:,1+(iSeries-1)*(nbd + naddb0) : iSeries*(nbd + naddb0)) = vnorm ; % Stack DW images in 4th dim
    


        % dscheme form getSeriesVERDICT is cell array with non-empty entries
        % only at SeriesNumbers where there is an XX file with IF_delta_Delta
        % (as SeriesNumbers typically jump in stesp of 100, this is largely
        % empty)

        %% GET SCHEME FROM B VALUES AND SCANNER (NEEDS ATTENTION FOR ALTERNATIVE SCHEMES)

        % % I suggest to just set scheme as loaded scheme here
        % if ~isempty(dscheme) && length(dscheme{sn_this}) > 1
        %     bvs = bv2scheme(bv_this, 'PDS', dscheme{sn_this}) ;
        %     nDeltaFromDicom  = nDeltaFromDicom  + 1 ;
        % else
        %     bvs = bv2scheme(bv_this, scanner) ; 
        % end
        % scheme(1+(iSeries-1)*(nbd+naddb0)+1 : iSeries*(nbd+naddb0)) = bvs ;
        % if naddb0 == 1
        %     scheme(1+(iSeries-1)*(nbd+naddb0)) = bv2scheme(0, scanner) ;
        % end
        


        % SET SCHEME AS EXPECTED SCHEME
        scheme = expected_scheme;
    
    
        % for output/debugging
        vprereg(:,:,:,1+(iSeries-1)*(nbd + 1) : iSeries*(nbd + 1)) = cat(4,vb0,vbv) ;
        vpostreg(:,:,:,1+(iSeries-1)*(nbd + 1) : iSeries*(nbd + 1)) = cat(4,vb0_reg,vbv_reg) ;
    
    
    
    end % iSeries



    %% Exclude b values from fitting

    % == Exclude b values from fitting!
    
    for bval = exclude_bvals
        
        disp(['b value ' num2str(bval) ' removed for fitting'])
    
        % Find scheme index for b value
        scheme_bools = (bvinY ~= bval);
    
        % FInd scheme indx for corresponding b=0
        [~, indx] = min(scheme_bools);
        scheme_bools(indx-1) = 0;

        % Exclude b value image from Y
        Y = Y(:,:,:,scheme_bools);
        bvinY = bvinY(scheme_bools);
    
    
    end


    % Some META data
    META = struct();
    META.highbval = max(vBV(:,2));
    META.excluded_bvals = exclude_bvals;
    META.schemename = opts.schemename;



    %% Create data folder and save relevant variables
    mkdir(datafolder)
    
    % LOAD NECESSARY VARIABLES
    save([datafolder '/dinfo.mat'], 'dinfo')
    save([datafolder '/geomout.mat'], 'geomout')
    save([datafolder '/Y.mat'], 'Y');
    save([datafolder '/scheme.mat'], 'scheme');
    save([datafolder '/b0fromhighb.mat'], 'b0fromhighb')
    save([datafolder '/META.mat'], 'META')


end




%% Add TE and TR to scheme
% Factors of 2 to account for added b=0!
for bindx = 1:length(scheme)/2

    bval = scheme(2*bindx).bval;
    
    % Find entries of dinfo matrix with tha bvalue
    bvalbools = ([dinfo.DiffusionBValue] == bval);

    % TE
    try
        TEs = [dinfo.EffectiveEchoTime];
    catch
        TEs = [dinfo.EchoTime];
    end
    bTEs = TEs(bvalbools);
    scheme(2*bindx-1).TE = bTEs(1);
    scheme(2*bindx).TE = bTEs(1);

    % TR
    TRs = [dinfo.RepetitionTime];
    bTRs = TRs(bvalbools);
    scheme(2*bindx-1).TR = bTRs(1);
    scheme(2*bindx).TR = bTRs(1);
end

% Update TEvector
TEvec = [scheme(1:2:end).TE];

% % if opts.register
% %     linkaxes(axrs)
% % end
% % 
% % hfmedian = figure(Name='medians', Visible=figVisible) ;
% % plot(plotbv,medianvb0,'LineWidth',2,'DisplayName','Median B0'), hold on
% % plot(plotbv,medianvbv,'LineWidth',2,'DisplayName','Median vbv')
% % plot(plotbv,medianvbv./medianvb0*medianvb0(1),'LineWidth',2,'DisplayName','Normalised median vbv')
% % grid on
% % xlabel('b-value'), ylabel('median')
% % legend
% % 
% % figrpt = Figure(hfmedian);
% % figrpt.Snapshot.Caption = 'Medians excluding end slices' ;
% % figrpt.Snapshot.ScaleToFit = true ;
% % append(rpt,figrpt);

% Display inputs Y for QA. Also place in report.

Dref = 2e-9 ;  % A reference diffusivity for scaling the signals

% When "extra" b=0 are in Y, this leads to unnecessary blanks displayed so
% remove from display. These are added at the start of each 'b-value'
if naddb0 == 0
    row2Y4 = 1:size(Y,4) ;
else
    addedRows = 1:(nbd+naddb0):size(Y,4) ;
    row2Y4 = 1:size(Y,4) ;
    row2Y4(addedRows) = [] ;
end
nRow = length(row2Y4) ;


%% ADAM ADDITIONS

% == Mask for fitting

% Check if mask has been inputted
if exist('opts.mask', 'var')
    disp('Input mask');

else % Bounding box mask (Not sure why input isn't working)
    disp('no mask')
    opts.mask = zeros([size(b0fromhighb,[1 2 3])]) ;
    opts.mask(44:132,44:132,:) = 1;
end


%% Apply fitting

% ===== PERFORM FITTING

% == ADC
if opts.calcADC
    bvVec = [scheme.bval];
    bvVecUseBools = (bvVec<opts.vADCbmax);

    bvVecUse = bvVec(bvVecUseBools);
    YUse = Y(:,:,:,bvVecUseBools);

    [ADC, Sb0] = calcADC(YUse, bvVecUse);

end


% == VERDICT

switch opts.fitting

    case 'VERDICT'

        switch opts.fittingtechnique
    
            case 'AMICO'
    
                [fIC, fEES, fVASC, R, rmse, A, tparams, vfopt] = verdict_fit( ...
                    scheme, ...
                    Y, ...
                    Rs=Rs, ...
                    solver=opts.solver, ...
                    ncompart=opts.ncompart, ...
                    mask = opts.mask) ;
    
            case 'MLP'
    
                [fIC, fEES, fVASC, R, rmse] = verdict_MLP_fit( ...
                    opts.schemename, ...
                    opts.modeltype, ...
                    Y, ...
                    noisetype = opts.noisetype,...
                    sigma0train = opts.sigma0train,...
                    T2train = opts.T2train,...
                    scheme = scheme,...
                    mask = opts.mask,...
                    schemesfolder = opts.schemesfolder,...
                    modelsfolder = opts.modelsfolder,...
                    pythonfolder = opts.pythonfolder...
                    );   
        end


    case 'RDI'

        switch opts.fittingtechnique
    
            case 'AMICO'
    
                [fIC, fEES, fVASC, R, rmse] = RDI_fit( ...
                    scheme, ...
                    Y, ...
                    Rs=Rs, ...
                    solver=opts.solver, ...
                    ncompart=opts.ncompart, ...
                    mask = opts.mask) ;
    
            case 'MLP'
    
                [fIC, fEES, fVASC, R, rmse] = verdict_MLP_fit( ...
                    opts.schemename, ...
                    opts.modeltype, ...
                    Y, ...
                    sigma0train = opts.sigma0train,...
                    scheme = scheme,...
                    mask = opts.mask,...
                    schemesfolder = opts.schemesfolder,...
                    modelsfolder = opts.modelsfolder,...
                    pythonfolder = opts.pythonfolder...
                    );   
        end

end



%% ADAM addition

% Small bit of code to make sure saved volumes are the correct orientation
dicomzs = [dinfo.ImagePositionPatient];
outputzs = [geomout.IPP];

if dicomzs(3,1) == outputzs(3, end)

    fIC = flip(fIC, 3);
    fEES = flip(fEES, 3);
    fVASC = flip(fVASC, 3);
    R = flip(R, 3);
    rmse = flip(rmse, 3);
    b0fromhighb = flip(b0fromhighb, 3);

    if opts.calcADC
        ADC = flip(ADC,3);
    end


end



% Save ADC
if opts.calcADC
    save([convertStringsToChars(output_folder) '/ADC.mat'], 'ADC')
end



% Save b0 from high b value
save([convertStringsToChars(output_folder) '/b0fromhighb.mat'], 'b0fromhighb')


end




function opts = convertCharsToLogical(opts) 
% convertCharsToLogical Converts a field with char value to 
% logical if 'true' or 'false'
% 

fields = fieldnames(opts) ;

for ifield = 1:length(fields)
    fname = fields{ifield} ;
    if ischar(opts.(fname))
        switch opts.(fname)
            case {'true', 'True','TRUE'}
                opts.(fname) = true ;
            case {'false','False','FALSE'}
                opts.(fname) = false ;
        end
    end
end

end

