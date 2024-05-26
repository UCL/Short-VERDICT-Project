function [vol_out, bval, bvec, sz_data, mask_locs] = dicom2noddi(varargin)
% DICOM2NODDI  Convert DICOM files to input for NODDI
%  Only extracts one slice. Allows user to draw ROI to restrict processing.
%
%  [vol_out, bval, bvec, sz_data, mask_locs] = dicom2noddi
%  [vol_out, bval, bvec, sz_data, mask_locs] = dicom2noddi(param, value, ...)
%
% vol_out  [npixels_in_roi ngrad]
%
% Parameter value pairs
%   'dinfo' :  structure from datparse that includes all relevant 
%              series for NODDI (if not present, will call datparse)
%
%   'slice' 
%
%   'mask_method' : 'roipoly', {'auto'}, 'full'
%
% Example
%
%  [sl, bval, bvec, sz_data, mask_locs] = ...
%    dicom2noddi('dinfo',dinfo, ...
%                  'd2mat_cell',{'series',2201,'slice', 10, 'op','fp' }) ;
% OR:
%
%  [sl_out, bval, bvec, sz_data, mask_locs] = dicom2noddi ;
%
% noddi_qa(sl_out, bval, bvec, sz_data, mask_locs)
%
%  noddi_batch_fitting(sl_out, bval, bvec, outputfile, sz_data, mask_locs)
%  noddi_params_view(outputfile)
%
% 
% David Atkinson  D.Atkinson@ucl.ac.uk
%
% See also  DATPARSE D2MAT NODDI_QA 
%


dinfo = [] ;

mask_method = 'auto' ;

for ip = 1:2:nargin
    switch varargin{ip}
        case 'dinfo'
            dinfo = varargin{ip+1} ;
        case {'slices', 'slice'}
            slices = varargin{ip+1} ;
        case 'mask_method'
            mask_method = varargin{ip+1} ;
        otherwise
            error(['Unknown parameter in input'])
    end
end

if isempty(dinfo)
    disp('Select ALL MultiFrame, or folder with all SingleFrame required')
    dinfo = datparse ;
end

top_sl = max([dinfo.sl]) ;
bot_sl = min([dinfo.sl]) ;

if ~exist('slices','var')
    slices = unique([dinfo.sl]);
end

nslice = length(slices) ;
sl_out_cell = cell([1 nslice]) ;

% Read in one slice at a time.
for islice = 1:nslice
    [slin, matp, sl_locs] = d2mat(dinfo,{'frame'}, 'slice', slices(islice),'op','fp') ;
    sz_sl = size(slin) ;
    nf = sz_sl(3) ;
    
    BWmask = ones(sz_sl(1), sz_sl(2)) ; % all pixels
    if islice == 1
        BWmask_vol = ones(sz_sl(1), sz_sl(2), nslice) ;
    end
    
    switch mask_method
        case 'roipoly'
            figure('Name','Select ROI')
            bord = iptgetpref('ImshowBorder');
            iptsetpref('ImshowBorder','tight')
            BWmask = roipoly(mat2gray(sum(slin,3))) ;
            iptsetpref('ImshowBorder',bord)
            BWmask_vol(:,:,islice) = BWmask ;
        case 'auto'
            th_img = sum(slin,3) ;
            thmax = prctile(th_img(:),95) ;
            jmsk = find(th_img > thmax/15) ;
            BWmask = zeros(size(th_img)) ;
            BWmask(jmsk) = 1 ;
            BWmask_vol(:,:,islice) = BWmask ;
            %figure('Name','masked image')
            %imshow(mat2gray(BWmask.*th_img))
        otherwise
            disp(['No mask applied. To recover the original:'])
            disp([' vol = reshape(vol_out, [sz_data size(vol,ndims(vol))]) ; '])
    end
    
    mask_locs_sl = find(BWmask==1) ;
    sl_out = zeros([length(mask_locs_sl) nf]) ;
    
    for iframe =1:nf
        tframe = slin(:,:,iframe) ;
        sl_out(:,iframe) = tframe(mask_locs_sl) ;
    end
    sl_out_cell{islice} = sl_out ;
    
end % end slice loop

mask_locs = find(BWmask_vol==1) ; % linear index into mask volume

% attempt below to prevent growing array problems
vol_out = zeros([length(mask_locs) nf]) ;
vol_out = sl_out_cell{1} ;
for islice =2:nslice
    vol_out = cat(1,vol_out,sl_out_cell{islice}) ;
end


bval = zeros([nf 1]) ;
bvec = zeros([nf 3]) ;

if ~isfield(dinfo,'DiffusionDirectionality')
    error(['No DiffusionDirectionality - needs MultiFrame DICOM at the moment'])
end

%hw = waitbar(0,['Processing ',num2str(nf),' frames']) ;
%wint = round(nf/100) ;

loc_strip = [] ;

for iframe = 1:nf
    
    bval(iframe) = dinfo(sl_locs(iframe)).DiffusionBValue ;
    
    switch dinfo(sl_locs(iframe)).DiffusionDirectionality
        case 0 % b=0
            bvec(iframe,:) = [0 0 0] ;
        case 1 % non-zero diffusion measurement
            bvec(iframe,:) = dinfo(sl_locs(iframe)).DiffusionGradientOrientation ;
        case 2 % trace image etc
            % Need to strip this entry from the data
            loc_strip = [loc_strip iframe] ;
        otherwise
            error(['Un recognised DiffusionDirectionality'])
    end
end
%close(hw)

vol_out(:,loc_strip) = [] ;
bval(loc_strip) = [] ;
bvec(loc_strip,:) = [] ;


sz_data = [sz_sl(1:2) nslice] ;







