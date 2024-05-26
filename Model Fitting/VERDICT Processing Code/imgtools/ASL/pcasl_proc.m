function [cbf, cbf_mask, params, dcalib, mcalib, locs_calib] = pcasl_proc
% PCASL_PROC  Processing for pCASL
%  [cbf, cbf_mask, params, dcalib, mcalib, locs_calib] = pcasl_proc 
%
% Example
% In DICOM select GUI, press Browse to choose folder containing Enhanced
% DICOM files.
% When it has finished updating, first select the main pCASL scan e.g. 
% 1001 EMR | WIP pCASL_main SENSE
% then the calibration e.g.
% 1101 EMR | WIP pCAS_calibration SENSE
% then close and cancel, or select the RAW file corresponding to the main e.g.
% 1001 RAW | WIP pCASL_main SENSE
%
%   [cbf, cbf_mask, params, dcalib, mcalib, locs_calib] = pcasl_proc ;
%   writeDicom(cbf.*cbf_mask, dcalib,locs_calib, 'pCASL','geom',mcalib.geom,'FrameOfReferenceUID','keep')
%
% D.Atkinson@ucl.ac.uk
% See also PCASL
%


cbf_disp_range = [10 100] ;

disp(['Select pCASL main scan'])
fn_main = dselect('ismodal',true) ;
dmain = datparse(fn_main) ; % Select main ASL file

disp(['Select pCASL calibration (PD) scan'])
fn_calib = dselect('ismodal',true) ;
dcalib = datparse(fn_calib) ; % Select calibration "PD" data
dcalib_info = dicominfo(dcalib(1).Filename) ;
switch dcalib_info.MagneticFieldStrength
    case 3
        params.T1_blood = 1650 ;
    case 1.5
        params.T1_blood = 1350 ;
    otherwise
        warning(['Unknown field strength'])
end

% Set other parameters
% read from raw data DICOM, if this fails, bring up dialog box
disp(['Cancel twice, or select DICOM RAW File corresponding to main pCASL scan:'])
disp(['   ',dmain(1).Filename,' with series number ',num2str(dmain(1).SeriesNumber)])
fn_xx = dselect('ismodal',true) ;
xxinfo = XXread(fn_xx) ;

if isfield(xxinfo,'EX_FLL_delay')
    params.pld = xxinfo.EX_FLL_delay;
end
if isfield(xxinfo,'EX_FLL_casl_dur')
    params.label_dur = xxinfo.EX_FLL_casl_dur ;
end

if ~isfield(params,'T1_blood') || ~isfield(params,'pld') || ~isfield(params,'label_dur')
    prompt = {'Enter T1_blood (ms):','Enter Post Label Delay (ms):','Enter Label Duration (ms)'};
    dlg_title = 'Input pCASL parameters';
    num_lines = 1;
    def = {'1650','2000','1650'};
    if isfield(params,'T1_blood') ; def{1} = num2str(params.T1_blood) ; end ;
    if isfield(params,'pld') ; def{2} = num2str(params.pld) ; end ;
    if isfield(params,'label_dur') ; def{3} = num2str(params.label_dur) ; end
    answer = inputdlg(prompt,dlg_title,num_lines,def);
    params.T1_blood = str2num(answer{1}) ;
    params.pld = str2num(answer{2}) ;
    params.label_dur = str2num(answer{3}) ;
end

[vasl, masl] = d2mat(dmain,{'slice','ilabtype','dyn'},'op','fp') ;
vasl = mean(vasl,5) ; % average dynamic
[vcalib, mcalib, locs_calib] = d2mat(dcalib,{'slice','dyn'},'op','fp') ;
vcalib = mean(vcalib,4) ;

[cbf, cbf_mask, params] = pcasl(vcalib, vasl(:,:,:,1), vasl(:,:,:,2), params) ;
params
eshow(cbf.*cbf_mask, mcalib.vdims)
cbfg = mat2gray(cbf.*cbf_mask,cbf_disp_range) ;
[X, map] = gray2ind(cbfg) ;
rgb = zeros([size(cbfg) 3]) ;
for islice = 1:size(cbfg,3)
    rgb(:,:,islice,:) = ind2rgb(X(:,:,islice),jet(length(map))) ;
end
eshow(rgb, 'vdim', mcalib.vdims, 'isrgb',1,'Name','CBF')

if nargout == 0 
    cbf = [] ;
end