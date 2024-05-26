function [vol, matp, locs_in] = d2mat(dinfo, outd, varargin)
% D2MAT  DICOM, PAR/REC or XMLPAR to matrix
%  [vol, matp, locs] = d2mat(dinfo, outd,   param, value, ...)
%
% dinfo a structure from DATPARSE (or DPARSE, DMFPARSE, PARPARSE)
%       dinfo can contain multiple series but for MultiFrame DICOM, do not
%       include more than required in output (e.g. no scout scans). May
%       fail if not all frames selected have same Width and Height.
% outd - a cell array listing the 3rd and subsequent output dimensions. 
%        (first two dims are ny nx). Can contain:
%   'slice','series','bv','dyn','bdirec','ddty','ctdt','wfio','fa','echo',
%   'itype', 'InversionTime', 'TinSeries', 'frame', 'effTE', 'aqno', 
%   'ilabtype', 'ASLphase'
% Note dyn reads TemporalPositionIdentifier
% Note 'ASLphase' can be specified as an output dimension, but not a param.
%
%  Note 'frame' is a special case, and vol will be [ ny nx nframe ]. In 
%  this case, you can only restrict by series or slice. See Examples. 
%  'frame' is useful if you are unsure of the data layout or struggling to
%  get exactly one loc, or if you need data in same order as in dinfo. Note
%  the order of data in dinfo may not be the temporal order of acqusition.
%
% param, value pairs to control output. These can restrict the output to a
% vector of specified values, e.g. specific slices. All outputs restricted 
% here MUST ALSO BE LISTED as dimensions of outd. If restricting to one 
% value, make this the last dimension of outd. 
% If an output is not specified, all instances will be output if they 
% match.  
% The data is pre-processd to include only the scans in 'series' (if
% present).
%
% ddty   (DiffusionDirectionality): 0 (NONE), 1 (DIRECTIONAL), 2 (ISOTROPIC)
% ctdt   CardiacTriggerDelayTime
% wfio   Dixon: 1 DixonWater, 2 DixonFat, 3 DixonInPhase, 4 DixonOutPhase
% echo   EchoNumber (Dixon MFFE)
% itype ImageType (duplicates wfio, set in data_process): 
%           1  'DERIVED\PRIMARY\W\W\DERIVED'
%           2  'DERIVED\PRIMARY\F\F\DERIVED'
%           3  'DERIVED\PRIMARY\IP\IP\DERIVED'
%           4  'DERIVED\PRIMARY\OP\OP\DERIVED'
%             
%           5  'ORIGINAL\PRIMARY\M_FFE\M\FFE'
%           6  'ORIGINAL\PRIMARY\M_IR\M\IR'
%           7  'ORIGINAL\PRIMARY\T2 MAP\T2\UNSPECIFIED'
%           8  'ORIGINAL\PRIMARY\R_IR\R\IR'
%           9  'ORIGINAL\PRIMARY\VELOCITY MAP\P\PCA'   B0 or velocity map
%           10 'ORIGINAL\PRIMARY\M_B1\M\B1'   B1 map (in percent)
%           11 'ORIGINAL\PRIMARY\M_SE\M\SE'
%           12 'ORIGINAL\PRIMARY\PHASE MAP\P\SE' Phase map
%           13 'ORIGINAL\PRIMARY\M_PCA\M\PCA' 
%           14 'ORIGINAL\PRIMARY\PHASE MAP\P\FFE'  Phase map from FFE (SWI)
%           15 'ORIGINAL\PRIMARY\PHASE MAP\P\B1'  Appears in B1 map if
%           phase output
%           16 'ORIGINAL\PRIMARY\PHASE MAP\P\IR'
%           17 'ORIGINAL\PRIMARY\R_FFE\R\FFE') ; % 17
%           18 'ORIGINAL\PRIMARY\I_IR\I\IR') ; % 18
%           19 'ORIGINAL\PRIMARY\I_FFE\I\FFE') ; % 19
%           20 'ORIGINAL\PRIMARY\B0 MAP\B0\UNSPECIFIED' Ingenia Single
%           Frame
%           21 'ORIGINAL\PRIMARY\T2_STAR_UNSPECIF\T2_STAR\UNSPECIFIED'
%           22 'DERIVED\PRIMARY\FF\FF\DERIVED'  Fat Fraction percentage (Ingenia?)
%
% ilabtype MRImageLabelType (ASL)
%      1  LABEL
%      2  CONTROL
%
% Inversion Time
%
% TinSeries  Acquisition Times in Series for multi-breath hold Siemens
% Single Frame, (No AcquisitionTime in file for Philips - use dyn) e.g.
%  [vol, mat] = d2mat(dinfo,{'slice','TinSeries','series'},'series',16,'op','fp') ;
%
% Effective Echo Time
%    [volte, mte] = d2mat(dinfoTE,{'slice','effTE'},'op','fp') ;
%     if there are other types in data (e.g. calc T2 map), instead see MET2
%
% Acquisition Number 'aqno'
%  Siemens PETMRI successive DSC blocks
%
% 'op' 
%     'pv' pixel value  - vol will be int16 DEFAULT
%     'pv_single' pixel value as a single
%     'dv' displayed value - DV = PV * RS + RI
%     'fp' floating point, allows for Philips scale slope  FP = DV / (RS * SS)
%
% 'resize'
%     rvec 2 element vector with new row and column size. Use NaN in one
%     element to preserve aspect ratio (see imresize. Now using 'lanczos3'
%     kernel.
%  
% OUTPUT
%  vol  - the matrix with size [ny nx ...] with the last dimensions
%  controlled by outd
%  matp, structure with possible fields: 
%   vdims - the voxel size along each dimension in mm or s (for dynamic). 
%            diffusion b-values, gradients, series numbers are all set to
%            1.
%   geom  - the geometry structure from DGEOMEXTRACT
%   bvVec - all b-values in dinfo, unless restricted 
%   bgrads - all diffusion gradient orientations in dinfo, unless restricted
%   If output is restricted by series, the XXX_indata refer only to data 
%   in the specified series, except for SeriesNumberVec_indata which is 
%   all the data in dinfo.
%   ctdtVec_indata    CardiacTriggerDelayTimes
%   wfioVec_indata    water,fat,inphase,outphase in data
%   fa_indata         flip angle in data
%   echo_indata
%   itype_indata
%   ti_indata       InversionTimes in data
%   effTEVec_indata    EffectiveEchoTimes in data
%
% locs
%   Locations in dinfo of the data used for this output. Output as a 
%   matrix with same size as vol, excluding the first two 
%  (row and column) dimensions. Useful for finding the original files, 
%   e.g for copying DICOM headers as in writeADC
%
% Example usage:
%   dinfo = datparse ;
%   [vol, matp] = d2mat(dinfo,{'slice','bv','series'}, ...
%                                'series',1001,'op','fp') ; 
% 
%   dinfo = datparse ;
%   [vol, matp, locs] = d2mat(dinfo,{'frame'}, 'series', 2201, 'op','fp') ;
%
% There has to be one and only one image that matches for the output
% matrix. For example if you specify only slice as output and there are
% multiple dynamics for the same slice position it will stop with an error. 
% Diffusion can be problematic with ADCs, trace images etc and the code 
% is moving towards using the DICOM Directionality flag.
%
% Diffusion
% A scan with b=0 and b>0 may have the individual directions
% in the files for the b>0 but not for the b=0. For Philips DWI, when
% the individual gradient directions are output, the DW have gradient 
% orientation indices 1:3 and the isotropic
% images have a gradient orientation index of 4, as does the b=0.
% For Philips DTI PAR, the b=0 image has a gradient orientation index of nd+1
% where nd is the number of directions in the DTI. For MultiFrame DICOM,
% see dmf2d.m or example below.
%
% DTI
% [volb0, matpall] = d2mat(dinfo,{'slice','bv'},'bv',0) ;
%   from a PAR, need to prevent the last "direction"
% [volD, matp] = d2mat(dinfo,{'slice','bdirec'},'bdirec',[1:size(matpall.bgrads,1)-1]) ;
% otherwise,
% [volD, matp] = d2mat(dinfo,{'slice','bdirec'}) ;
% Multi-Frame DICOM
%  [volD, matp] = d2mat(dinfo,{'slice','bdirec','ddty'},'ddty',1,'op','fp') ;
%  [volb0, matb0] = d2mat(dd2,{'slice','bv','ddty'},'bv',0,'ddty',0,'op','fp') ;
%  D = dwi2tensor(volD, matp.bgrads, matp.bvVec_indata(end), volb0) ;
%  eshow(1e6*invariantsb(D,'mean'),'name','mean')
%
% DWI with individual diffusion directions in the file
% [volb0, matp_all] = d2mat(dinfo,{'slice','bv'},'bv',0) ;
% [volisoD, matp] =d2mat(dinfo,{'slice','bv','bdirec'},'bdirec',[size(matp_all,1)]) ;
% bvs = unique([dinfo.DiffusionBValue]) ;
% volnonb0 = d2mat(dinfo,{'slice','bv','bdirec'},'bv',bvs(2:end)) ;
% 
%
% DIXON
%   MFFE source echoes
%     d2mat(dinfoee,{'slice','fa','echo'},'echo',1,'op','fp') ;
%   Water/Fat/In Phase /OutPhase
%     [vol, matp] = d2mat(dinfo,{'slice','fa','wfio'},'wfio',[3],'op','fp') ;
%     (for in phase)
% B1
% If required to separate from M_IR or T2_MAP
% On SingleFrame data from Ingenia, B1 maps have itype 10. On MultiFrame
% from Achieva, use itype 7
%
% B0 maps
% Ingenia R517 B0 maps have itype 20
%  
% Dynamics
%  Philips multiframe Dixon
%    [vol, mat, locs] = d2mat(dinfo,{'slice','dyn','wfio'},'wfio',1,'op','fp') ;
%
% QFlow
%  Philips QFlow.  
%   dflow = datparse ;
%   [vf, mf] = d2mat(dflow,{'ctdt','itype','slice'},'itype',9,'op','dv') ;
%
% ASL
%  See pcasl.m (essentially read slice, dyn and ilabtype)
%  MJS added code for Single Frame DICOMs, example use:
%     [vasl, masl] = d2mat(dinfo,{'slice','ilabtype','dyn','ASLphase'},'op','fp') ;
%      % Currently you cannot have 'ASLphase' in the param/value pairs,
%      i.e. you cannot restict the output
%
% Copyright, 2019, David Atkinson
% D.Atkinson@ucl.ac.uk
%
% See also DATPARSE DPARSE XMLPARSE PARPARSE DMFPARSE DGEOMEXTRACT
%

% 
%#  PV = pixel value in REC file, FP = floating point value, DV = displayed value on console
%#  RS = rescale slope,           RI = rescale intercept,    SS = scale slope
%#  DV = PV * RS + RI             FP = DV / (RS * SS)


nf = length(dinfo) ;

% Set defaults
restrictOnSeries = true ;


SeriesNumberVec = [dinfo.SeriesNumber] ;
SeriesNumberVec = unique(SeriesNumberVec) ;
matp.SeriesNumberVec_indata = SeriesNumberVec ;

SliceVec = [dinfo.sl] ;
SliceVec = unique(SliceVec) ;
matp.SliceVec_indata = SliceVec ;

% Determine if MultiFrame one file or multiple selected
if isfield(dinfo(1),'Frame')
    MFrame = true ; % MultiFrame DICOM
    fns = {dinfo.Filename} ;
    [ufns, ifns] = unique(fns) ;
    if length(ufns) == 1
        % dmfinfo = dicominfo(dinfo(1).Filename) ;
        mMfile = false ; % Not multiple MultiFrame files (i.e. one file)
        
    else
        mMfile = true ;
        sort_ifns  = sort(ifns) ;
        Vpre = zeros([dinfo(1).Height dinfo(1).Width 1 length(dinfo)],'uint16') ;
        vstart = 1 ;
        frame_offset =0 ;
       
        for imMfile = 1:length(sort_ifns)
            temp = dicomread(fns{sort_ifns(imMfile)}) ;
            Vpre(:,:,1,vstart:vstart+size(temp,4)-1) = temp ;
            for istruc = vstart:vstart+size(temp,4)-1
              dinfo(istruc).cumFrame = dinfo(istruc).Frame + frame_offset ;
            end
            vstart = vstart + size(temp,4)  ;
            frame_offset = frame_offset + size(temp,4) ;
        end
        clear temp
    end
    
else
    MFrame = false ;
end

% Restrict on series to prevent properties in unintersting series
% confounding code.
loc_sn = true([1 nf]) ;
loc_sl = true([1 nf]) ;
loc_ext =  true([1 nf]) ; % locations extracted here
if restrictOnSeries
    nv = length(varargin) ;
    for ip = 1:2:nv
        switch varargin{ip}
            case { 'series'}
                SeriesNumberVec = varargin{ip+1} ;
                sn_all = [dinfo.SeriesNumber] ;
                loc_sn = ismember(sn_all, SeriesNumberVec) ;
            case { 'slice' }
                SliceVec = varargin{ip+1} ;
                sl_all = [dinfo.sl] ;
                loc_sl = ismember(sl_all, SliceVec) ;
        end
    end
    loc_ext = loc_sn & loc_sl ; % logical AND
    dinfo = dinfo(loc_ext) ;
    loc_ext_orig = loc_ext ;
    loc_ext = [1:length(dinfo)] ; %????
    
    nf = length(dinfo) ;
    SeriesNumberVec = [dinfo.SeriesNumber] ;
    SeriesNumberVec = unique(SeriesNumberVec) ;
    SliceVec = [dinfo.sl] ;
    SliceVec = unique(SliceVec) ;
    
end

% Allocate sframe to dinfo to enable frame option. (Note Frame is a field
% for MultiFrame DICOM so use field name sframe here.

for iframe = 1:length(dinfo)
    dinfo(iframe).sframe = iframe ;
end
sframeVec = [dinfo.sframe];



if isfield(dinfo,'DiffusionBValue') 
    bvVec = [dinfo.DiffusionBValue] ;
    bvVec = unique(bvVec) ;
    matp.bvVec_indata = bvVec ;
    if length(SeriesNumberVec) > 1
      dinfo = fillstruct(dinfo, 'DiffusionBValue', NaN) ;
    end
end

% Dixon
if isfield(dinfo, 'wfio')
    wfioVec = [dinfo.wfio] ;
    wfioVec = unique(wfioVec) ;
    matp.wfioVec_indata = wfioVec ;
end

if isfield(dinfo, 'TinSeries')
    TinSeriesVec = [dinfo.TinSeries] ;
    TinSeriesVec = unique(TinSeriesVec) ;
    matp.TinSeriesVec_indata = TinSeriesVec ;
end

if isfield(dinfo, 'AcquisitionNumber')
    AqnoVec = [dinfo.AcquisitionNumber] ;
    AqnoVec = unique(AqnoVec) ;
    matp.AqnoVec_indata = AqnoVec ;
end


% ImageType and ImageLabelType Enumerations are in data_process.m
if isfield(dinfo, 'itype')
    itypeVec = [dinfo.itype] ;
    itypeVec = unique(itypeVec) ;
    matp.itypeVec_indata = itypeVec ;
end

if isfield(dinfo, 'ilabtype')
    ilabtypeVec = [dinfo.ilabtype] ;
    ilabtypeVec = unique(ilabtypeVec) ;
    matp.ilabtypeVec_indata = ilabtypeVec ;
end

% Inversion Time
if isfield(dinfo, 'InversionTime')
    tiVec = [dinfo.InversionTime] ;
    tiVec = unique(tiVec) ;
    matp.tiVec_indata = tiVec ;
end

% Effective Echo Times
if isfield(dinfo, 'EffectiveEchoTime')
    effTEVec = [dinfo.EffectiveEchoTime] ;
    effTEVec = unique(effTEVec) ;
    matp.effTEVec_indata = effTEVec ;
end

if isfield(dinfo, 'EchoNumber')
    echoVec = [dinfo.EchoNumber] ;
    echoVec = unique(echoVec) ;
    matp.echoVec_indata = echoVec ;
end

if isfield(dinfo, 'EchoNumbers')  % extra 's'
    echoVec = [dinfo.EchoNumbers] ;
    echoVec = unique(echoVec) ;
    matp.echoVec_indata = echoVec ;
end

if isfield(dinfo, 'EchoTime') 
    echoTimeVec = [dinfo.EchoTime] ;
    echoTimeVec = unique(echoTimeVec) ;
    matp.echoTimeVec_indata = echoTimeVec ;
end

if isfield(dinfo, 'FlipAngle')
    faVec = [dinfo.FlipAngle] ;
    faVec = unique(faVec) ;
    matp.faVec_indata = faVec ;
end

% Cardiac Trigger Delay Time (used for Look Locker)
% If from Single Frame, copy over Trigger Time
if ~isfield(dinfo,'CardiacTriggerDelayTime') && isfield(dinfo,'TriggerTime')
    [dinfo(:).CardiacTriggerDelayTime] = dinfo.TriggerTime ;
end

if isfield(dinfo,'CardiacTriggerDelayTime')
    ctdtVec = [dinfo.CardiacTriggerDelayTime];
    ctdtVec = unique(ctdtVec) ;
    matp.ctdtVec_indata = ctdtVec ; 
end

% MJS added ASLphase information Private_2001_1008 and Private_2001_1017 
if isfield(dinfo, 'Private_2001_1008')
    aslphaseNumberVec = [dinfo.Private_2001_1008] ;
    aslphaseNumberVec = unique(aslphaseNumberVec) ;
    matp.aslphaseNumberVec_indata = aslphaseNumberVec ;
end
 
if isfield(dinfo, 'Private_2001_1017')
    numASLPhases = [dinfo.Private_2001_1017] ;
    numASLPhases = unique(numASLPhases) ;
    matp.numASLPhases_indata = numASLPhases ;
end

if isfield(dinfo,'TemporalPositionIdentifier') 
    DynVec = unique([dinfo.TemporalPositionIdentifier]) ;
    matp.DynVec_indata = DynVec ;
end

if isfield(dinfo,'DiffGradOrientIdentifier') 
    bgVec = unique([dinfo.DiffGradOrientIdentifier]) ;
    
    % set bgall (note DiffGradOrientIdentifier might not be set in all
    % frames of MultiFrame DICOM
    ldinfo = length(dinfo) ;
    bgall = zeros([1 ldinfo]) ;
    for idinf = 1:ldinfo
        if length(dinfo(idinf).DiffGradOrientIdentifier) == 0
            bgall(idinf) = NaN ;
        else
            bgall(idinf) = dinfo(idinf).DiffGradOrientIdentifier ;
        end
    end
    
end

if isfield(dinfo,'DiffusionDirectionality')
    DdtyVec = unique([dinfo.DiffusionDirectionality]) ;
end

op = 'pv' ; % output pixel value
resize = false ;


nv = length(varargin) ;
for ip = 1:2:nv
    switch varargin{ip}
        case { 'series'}
            SeriesNumberVec = varargin{ip+1} ;
        case {'frame '}
            sframeVec = varargin{ip+1} ;
        case {'Slice', 'slice', 'Slices', 'slices'}
            SliceVec = varargin{ip+1} ;
        case {'bvalue','bvalues','bv'}
            if ~exist('bvVec','var')
                warning(['No diffusion in input scans'])
            end
            bvVec = varargin{ip+1} ;
         case {'bdirec'}
            if ~exist('bgVec','var')
                warning(['No diffusion grad orientations in input scans'])
            end
            bgVec = varargin{ip+1} ; 
         case {'dyn'}
            if ~exist('DynVec','var')
                warning(['No dynamics in input scans'])
            end
            DynVec = varargin{ip+1} ;  
        case {'ddty','DiffusionDirectionality'}
            if ~exist('DdtyVec','var')
                warning(['DiffusionDirectionality not in input scan.'])
            end
            DdtyVec = varargin{ip+1} ; 
        case {'ctdt'}
            if ~exist('ctdtVec','var')
                warning(['CardiacTriggerDelayTime not in input scan.'])
            end
            ctdtVec = varargin{ip+1} ;
        case {'wfio'}
            if ~exist('wfioVec','var')
                warning(['Dixon water fat in out not in input data'])
            end
            wfioVec = varargin{ip+1} ;
        case {'itype'}
            if ~exist('itypeVec','var')
                warning(['Image Type (itype) not in input data'])
            end
            itypeVec = varargin{ip+1} ;
        case {'ilabtype'}
            if ~exist('ilabtypeVec','var')
                warning(['Image Label Type (ilabtype) not in input data'])
            end
       case {'InversionTime'}
            if ~exist('tiVec','var')
                warning(['Inversion Time not in input data'])
            end
            tiVec = varargin{ip+1} ;
      case {'effTE'}
            if ~exist('effTEVec','var')
                warning(['Effective Echo Time not in input data'])
            end
            effTEVec = varargin{ip+1} ;
        case {'aqno'}
            if ~exist('AqnoVec','var')
                warning(['Acquisition Number not in input data'])
            end
            AqnoVec = varargin{ip+1} ;
        case {'echo'}
            if ~exist('echoVec','var')
                warning(['Echo numbers not in input data'])
            end
            echoVec = varargin{ip+1} ;
        case {'TinSeries'}
            if ~exist('TinSeriesVec','var')
                warning(['TinSeries not in input data'])
            end
            TinSeriesVec = varargin{ip+1} ;
        case {'fa'}
            if ~exist('faVac','var')
                warning(['Flip angles not in input data'])
            end
            faVec = varargin{ip+1} ;
        case {'op'}
            op = varargin{ip+1} ;
        case {'resize'}
            rvec = varargin{ip+1} ;  
            resize = true ;
        otherwise
            warning(['Unknown Parameter: ',varargin{ip}])
    end
end

if length(SeriesNumberVec) > 1
    warning(['More than one series'])
end

% loop over outd
% case slice, dyn etc
% set sz in each dimension
nd = length(outd) ;
sz = ones([1 4]) ;
loc1 = [1:nf] ; loc2 = [1:nf]; loc3 = [1:nf]; loc4=[1:nf];


sl_inop = false ;

for ioutd = 1:nd
    switch outd{ioutd}
        case 'slice'
            sl_inop = true ; vdsl = [] ;
            dp(ioutd).vals = [dinfo.sl] ;
            dp(ioutd).vec = SliceVec ;
            matp.SliceVec = SliceVec ;
            
            ioutd_sl = 2+ioutd ; % needed for setting vdims
            % !! Below can be wrong for more than one series in input.
            % vdims(2+ioutd) = dinfo(1).slc2c ;
        case 'series'
            dp(ioutd).vals = [dinfo.SeriesNumber] ;
            dp(ioutd).vec = SeriesNumberVec ;
            matp.SeriesNumberVec  = SeriesNumberVec ;
            vdims(2+ioutd) = 1;
        case 'frame'
            dp(ioutd).vals = [dinfo.sframe] ;
            dp(ioutd).vec = sframeVec ;
            vdims(2+ioutd) = 1 ;
            if ioutd ~= 1
                warning(['frame should be first outd dimension specified'])
            end
        case 'dyn'
            dp(ioutd).vals = [dinfo.TemporalPositionIdentifier] ;
            dp(ioutd).vec = DynVec ;
            matp.DynVec = DynVec ;
            
            if isfield(dinfo,'AcquisitionTime')
                aqt = [dinfo.AcquisitionTime] ;
                aqtu = unique(aqt) ;
                if length(aqtu) == 1
                    vdims(2+ioutd) = 0 ;
                else
                    diffaqt = diff(aqtu) ;
                    udiff = unique(diffaqt) ;
                    if length(udiff) > 1
                        warning(['Acquisition Time steps differ, using first'])
                        disp([' steps are: ',num2str(udiff)])
                        
                        vdims(2+ioutd) = udiff(1) ;
                    else
                        vdims(2+ioutd) = udiff ;
                    end
                end
            else
                vdims(2+ioutd) = 1 ;
            end
            
            
            
        case 'bv'
            dp(ioutd).vals = [dinfo.DiffusionBValue] ;
            dp(ioutd).vec = bvVec ;
            matp.bvVec = bvVec ;
            vdims(2+ioutd) = 1;
        case 'bdirec'
            dp(ioutd).vals = bgall ;
            dp(ioutd).vec = bgVec ;
            matp.bgVec = bgVec ;
            vdims(2+ioutd) = 1;
        case 'ddty'
            dp(ioutd).vals = [dinfo.DiffusionDirectionality] ;
            dp(ioutd).vec = DdtyVec ;
            matp.DdtyVec = DdtyVec ;
            vdims(2+ioutd) = 1 ;
        case 'wfio'
            dp(ioutd).vals = [dinfo.wfio] ;
            dp(ioutd).vec = wfioVec ;
            matp.wfioVec = wfioVec ;
            vdims(2+ioutd) = 1;
       case {'TinSeries'}
            dp(ioutd).vals = [dinfo.TinSeries] ;
            dp(ioutd).vec = TinSeriesVec ;
            matp.TinSeriesVec = TinSeriesVec ;
            vdims(2+ioutd) = 1;
       case 'itype'
            dp(ioutd).vals = [dinfo.itype] ;
            dp(ioutd).vec = itypeVec ;
            matp.itypeVec = itypeVec ;
            vdims(2+ioutd) = 1;
        case 'ilabtype'
            dp(ioutd).vals = [dinfo.ilabtype] ;
            dp(ioutd).vec = ilabtypeVec ;
            matp.ilabtypeVec = ilabtypeVec ;
            vdims(2+ioutd) = 1;
        % MJS added cases for ASLphase:     
        case 'ASLphase'
            dp(ioutd).vals = [dinfo.Private_2001_1008] ;
            dp(ioutd).vec = aslphaseNumberVec ;
            matp.aslphaseNumberVec = aslphaseNumberVec ;
            vdims(2+ioutd) = 1; 
       case 'InversionTime'
            dp(ioutd).vals = [dinfo.InversionTime] ;
            dp(ioutd).vec = tiVec ;
            matp.tiVec = tiVec ;
            vdims(2+ioutd) = 1;
       case 'effTE'
            dp(ioutd).vals = [dinfo.EffectiveEchoTime] ;
            dp(ioutd).vec = effTEVec ;
            matp.effTEVec = effTEVec ;
            vdims(2+ioutd) = 1;
        case 'aqno'
            dp(ioutd).vals = [dinfo.AcquisitionNumber] ;
            dp(ioutd).vec = AqnoVec ;
            matp.AqnoVec = AqnoVec ;
            vdims(2+ioutd) = 1;
        case 'echo'
            if isfield(dinfo, 'EchoNumber')
                dp(ioutd).vals = [dinfo.EchoNumber] ;
            else
                dp(ioutd).vals = [dinfo.EchoNumbers] ;
            end
            dp(ioutd).vec = echoVec ;
            matp.echoVec = echoVec ;
            vdims(2+ioutd) = 1;
        case 'fa'
            dp(ioutd).vals = [dinfo.FlipAngle] ;
            dp(ioutd).vec = faVec ;
            matp.faVec = faVec ;
            vdims(2+ioutd) = 0;

        case 'ctdt'
            dp(ioutd).vals = [dinfo.CardiacTriggerDelayTime] ;
            dp(ioutd).vec = ctdtVec ;
            matp.ctdtVec = ctdtVec ;
            
            aqtu = ctdtVec ;
            if length(aqtu) == 1
                vdims(2+ioutd) = 0 ;
            else
                diffaqt = diff(aqtu) ;
                udiff = unique(diffaqt) ;
                if length(udiff) > 1
                    warning(['CardiacTriggerDelayTime steps differ, using first'])
                    disp([' steps are: ',num2str(udiff)])
                    
                    vdims(2+ioutd) = udiff(1) ;
                else
                    vdims(2+ioutd) = udiff ;
                end
            end
            
        otherwise
            warning(['Unknown outd param: ',outd{ioutd}])
    end
    
    sz(ioutd) = length(dp(ioutd).vec) ;
end


firstpass = 1 ;

if MFrame && ~mMfile
  dmfinfo = dicominfo(dinfo(1).Filename) ;
  Xall = dicomread(dmfinfo) ;
end

locs = zeros([sz(1) sz(2) sz(3) sz(4)],'uint32') ;

for i1 = 1:sz(1)
    if nd >=1  
        loc1 = find(dp(1).vals == dp(1).vec(i1)) ;
    end
    
    for i2 = 1:sz(2)
        if nd >= 2  
          loc2 = find(dp(2).vals == dp(2).vec(i2)) ;
        end
        for i3 = 1:sz(3)
            if nd >= 3
              loc3 = find(dp(3).vals == dp(3).vec(i3)) ;
            end
            for i4=1:sz(4)
                if nd >= 4
                  loc4 = find(dp(4).vals == dp(4).vec(i4)) ;
                end
                
                loc = intersect(loc1,loc2) ;
                loc = intersect(loc, loc3) ;
                loc = intersect(loc, loc4) ;
                
                if length(loc) == 0
                    warning(['Found no slices to match current parameter set.'])
                    matp
                    disp([' If diffusion, try:'])
                    disp(['   cat(1,[dinfo.DiffusionBValue]'',[dinfo.DiffusionDirectionality]'' )'])
                    error(['Found no slices to match current parameter set.'])
                end
                if length(loc) > 1
                    matp
                    disp(['Restrict output so that there is exactly ',...
                        'one file for each entry in vol'])
                    disp([' e.g. restrict b-values or to scans (''series'') required'])
                    disp(['Could try dicominfo(''',dinfo(loc(1)).Filename,''')'])
                    disp([' '])
                    disp(['The dimensions of vol are [ny nx nd1 nd2 ...]'])
                    disp([' where nd1 is the size of parameter 1 etc in '])
                    disp([' d2mat(dinfo,{''param1'',''param2'', ...})'])
                    
                    error(['Found: ',num2str(length(loc)),... 
                        ' locs. Should be exactly 1'])
                end
                
                if firstpass
                    vdims(1) = dinfo(loc).PixelSpacing(1) ;
                    vdims(2) = dinfo(loc).PixelSpacing(2) ;
                    matp.vdims = vdims; % note matp.vdims can be rescaled below if image is resized
                else
                    if abs(vdims(1)-dinfo(loc).PixelSpacing(1)) > 0.001 || ...
                          abs(vdims(2)-dinfo(loc).PixelSpacing(2)) > 0.001 
                      warning(['Pixel Dimensions not all the same. vdims: ', ...
                          num2str(vdims(1:2)), 'PixelSpacing: ',num2str(dinfo(loc).PixelSpacing')])
                    end
                end
                
                if isfield(dinfo(loc),'RecFileName')
                    X = xmlparrecread(dinfo(loc)) ; % PAR or XML formats
                elseif isfield(dinfo(loc),'Frame') 
                    % Multi-frame DICOM
                    if mMfile 
                        % multiple MultiFrame files
                        if exist('Vpre')
                            X = Vpre(:,:,1,dinfo(loc).cumFrame) ;
                        else
                          X = dicomread(dinfo(loc).Filename,'frames',dinfo(loc).Frame) ;
                        end
                    else
                        % Single MultiFrame file
                        % was just loc at the end, but was wrong when series of
                        % slice restriction shortens dinfo
                        
                        X = Xall(:,:,1,dinfo(loc).Frame) ;
                        % X = dicomread(dmfinfo,'frames',dinfo(loc).Frame) ;
                    end
                else
                    X = dicomread(dinfo(loc).Filename) ;
                end
                
                if sl_inop
                    vdsl = unique([vdsl dinfo(loc).slc2c]) ;
                end
                
                switch op
                    case 'pv'
                        % pixel values - do nothing
                    case 'pv_single'
                        X = single(X) ;
                    case 'dv'
                        X = single(X)*dinfo(loc).RescaleSlope + ...
                                          dinfo(loc).RescaleIntercept ;  
                    case 'fp'
                        X = (single(X)*dinfo(loc).RescaleSlope + ...
                                          dinfo(loc).RescaleIntercept) / ...
                           (dinfo(loc).RescaleSlope * dinfo(loc).Private_2005_100e) ;
                    otherwise
                        warning(['Unknown op: ',op])
                end 
                
                if resize
                    if firstpass
                        [Ny,Nx,Nall] = size(X);
                        X = imresize(X,rvec,'lanczos3') ;
                        [Nyr,Nxr,Nall] = size(X);
                        matp.vdims(1) = matp.vdims(1) * Ny/Nyr ;
                        matp.vdims(2) = matp.vdims(2) * Nx/Nxr ;
                    else
                      X = imresize(X,rvec,'lanczos3') ;
                    end
                end
                
                if firstpass %pre-allocate space for vol
                    firstpass = 0 ;
                    vol = zeros([size(X,1) size(X,2) sz(1) sz(2) sz(3) sz(4)],class(X)) ;
                end
                vol(:,:,i1,i2,i3,i4) = X ;
                locs(i1,i2,i3,i4) = loc ;
            end
        end
    end
end

if sl_inop
    if length(vdsl) ~= 1
        warning(['Unable to set slice thickness in vdims'])
        disp(['Length(vdsl): ',num2str(length(vdsl))])
        matp.vdims(ioutd_sl) = NaN;
    else
      matp.vdims(ioutd_sl) = vdsl ;
    end
end

% In MultiFrame DICOM DiffGradOrientIdentifier will be empty for some
% frames so do not use:   bgall = [dinfo.DiffGradOrientIdentifier] ;
if exist('bgVec','var')
    bgrads = zeros([length(bgVec) 3]) ;
    
    for ibg = 1:length(bgVec)
        loc = find(bgall == bgVec(ibg)) ;
        bgrads(ibg,:) = dinfo(loc(1)).DiffusionGradientOrientation ;
    end

    matp.bgrads = bgrads ;
end

% set geom structure
geom = dgeomextract(dinfo(locs(:))) ;
matp.geom = geom ;

% if isempty(loc_ext)
%     locs_in = locs ;
% else
%     nzlocs = find(loc_ext) ;
%     locs_in = nzlocs(locs) ;

nzlocs = find(loc_ext_orig) ;
    locs_in = nzlocs(locs) ;
    locs_in = reshape(locs_in,size(locs)) ;
% end

if isfield(dinfo,'RepetitionTime')
  matp.RepetitionTime_Vec = unique([dinfo(locs).RepetitionTime] ) ;
end

% For DV output report on hidden scale factors
switch op
    case 'dv'
        %%ss = [dinfo(locs_in).Private_2005_100e ] ;
        ss = [dinfo(locs).Private_2005_100e ] ;
        uq_ss = unique(ss) ;
        if length(uq_ss) ==1 
            if uq_ss ~= 1
              disp(['A single private scale factor was ignored, value: ', ...
                  num2str(uq_ss)])
            end
        else
            warning(['Data contains different Private Scale factors that have not been applied'])
            disp(['You selected ''dv'' for output, ''fp'' would apply scales in'])
            disp(['range: ',num2str(min(ss)),' to ',num2str(max(ss))])
        end
end
        
disp(['d2mat: Output vol has size: ',num2str(size(vol)),' (',class(vol),')'])
end
                
    

    
    