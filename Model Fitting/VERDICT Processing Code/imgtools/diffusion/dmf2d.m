function [D, op] = dmf2d(varargin)
% DMF2D DICOM MultiFrame to Diffusion Tensor
% !! WARNING !! Slices are read in the order specified by the DICOM slice
% number, there is no check here on maintaining a right-handed coordinate
% system. 
%
%  [D, op] = dmf2d(param, value, ...)
%
% Parameter Value Pairs
%  'dinfo'  info structure from datparse for MultiFrame DICOM files.
%  'slices'  {'mid'} , 'all', [vec] slices to process (defaults to single mid-slice)
%  'B0_source'  'fit' | {'data'}
%
% op is a structure with fields, dinfo, B0, bvec, bval, mat and loc1
%   mat is the parameters read using d2mat, loc1 is a representative frame
%   in the original DICOMs used to gain header info for writeDicom
%
% Examples
%  D = dmf2d ;
%
% OR
%  dinfo = datparse(dselect) ;
%  [D, op] = dmf2d('dinfo',dinfo,'slices','all') ;
%
%  cfa = d2cfa(D) ;
%  writeDicom(cfa,'rgb','folder_name',folder_name, 'FrameOfReferenceUID' , 'keep', ...
%    'header' ,{dinfo, op.loc1(:)} ,'geom', [op.mat.geom])
%
% eshow(cfa,'isrgb',1,'name','cfa')
% eshow(1-invariantsb(D,'fa'),'name','1-FA')
% eshow(1e6*invariantsb(D,'mean'),'name','mean')
%
% eshow(op.B0,'name','S0')
%
% See for example d2radial to compute radial and axial 
%  [dradial, daxial] = d2radial(D) ;
% eshow(1e6*dradial,'name','Radial Diffusivity')
% eshow(1e6*daxial,'name','Axial Diffusivity')
%
% David Atkinson  D.Atkinson@ucl.ac.uk
% See also dwi2tensor dicom2noddi
%

%default
B0_source = 'data' ;  %B0 will come from data and not be fitted in dwi2tensor

npv = length(varargin) ;
for ipv = 1:2:npv
   switch varargin{ipv}
       case 'dinfo'
           dinfo = varargin{ipv+1} ;
       case {'slices'}
           slices = varargin{ipv+1} ;
       case 'B0_source'
           B0_source = varargin{ipv+1} ;
       otherwise
           error(['Unknown input parameter: ',varargin{ipv}])
   end
end

if ~exist('dinfo','var')
    disp(['Select all required MultiFrame Diffusion scans '])
    dinfo = datparse ;
end

height = unique([dinfo.Height]);
width  = unique([dinfo.Width]) ;
if length(height)~=1 || length(width)~=1
    error(['All images should be same size'])
end

if ~isfield(dinfo,'DiffusionDirectionality')
     error(['No DiffusionDirectionality - needs MultiFrame DICOM'])
end
    
if ~exist('slices','var')
    slices = ceil(max([dinfo.sl])/2) ;
else
    if ischar(slices)
        switch slices
            case 'mid'
                slices = ceil(max([dinfo.sl])/2) ;
            case 'all'
                slices = [1: max([dinfo.sl]) ];
            otherwise
                error(['Unknown slices option: ',slices])
        end
    else
        % slices is correct vector
    end
end

nslice = length(slices) ;
D = zeros(height,width,nslice,3,3) ;
B0 = zeros(height, width, nslice) ;


for islice = 1:nslice
    
    [vol, matp, locs] = d2mat(dinfo,{'frame'},'slice', slices(islice), 'op','fp') ;
    if islice == 1
        mat = matp ;
    else
        mat(islice) = matp ;
    end
    
    nf = size(vol,3) ;
    bval = zeros([nf 1]) ;
    bvec = zeros([nf 3]) ;
    
    
    hw = waitbar(0,['Processing ',num2str(nf),' frames']) ;
    wint = round(nf/100) ;
    
    loc_strip = [] ;
    
    for iframe = 1:nf
        if rem(nf,wint)== 0
            waitbar(iframe/nf,hw)
        end
        %tframe = vol(:,:,iframe) ;
        %sl_out(:,iframe) = tframe(mask_locs) ;
        
        bval(iframe) = dinfo(locs(iframe)).DiffusionBValue ;
        
        switch dinfo(locs(iframe)).DiffusionDirectionality
            case 0 % b=0
                bvec(iframe,:) = [0 0 0] ;
            case 1 % non-zero diffusion measurement
                bvec(iframe,:) = dinfo(locs(iframe)).DiffusionGradientOrientation ;
            case 2 % trace image etc
                % Need to strip this entry from the data
                loc_strip = [loc_strip iframe] ;
            otherwise
                error(['Un recognised DiffusionDirectionality'])
        end
    end
    close(hw), drawnow
    
    bval(loc_strip) = [] ;
    bvec(loc_strip,:) = [] ;
    vol(:,:,loc_strip) = [] ;
    
    locs_rep = locs;
    locs_rep(loc_strip) = [] ;
    loc1(islice) = locs_rep(1) ; % representative loc for writeDicom
    
    jb0 = find(bval==0);
    jnon0 = find(bval > 0) ;
    
    if strcmp(B0_source,'data') && isempty(jb0)
        warning(['No B0 in data, fitting to bvalues in data'])
        B0_source = 'fit' ;
    end
    
    switch B0_source
        case 'data'
            volb0 = vol(:,:,jb0) ;
            B0(:,:,islice) = sum(volb0,3) / size(volb0,3) ;
            
            volb  = vol(:,:,jnon0) ;
            bv = bval(jnon0) ;
            grad = bvec(jnon0,:) ;
            
            ny = size(volb,1) ; nx = size(volb,2) ; nz = 1 ;
            ngrad = length(bv) ;
            
            volb = reshape(volb,[ny nx nz ngrad]) ;
            
            D(:,:,islice,:,:) = dwi2tensor(volb, grad, bv, B0(:,:,islice)) ;
            
        case 'fit'
            [D(:,:,islice,:,:),B0(:,:,islice)] = dwi2tensor(volb, bvec, bval) ;
        otherwise
            error(['Unknown B0_source value: ',B0_source])
    end
end

if nargout > 1
    op.dinfo = dinfo ;
    op.B0 = B0 ;
    op.bvec = bvec ;
    op.bval = bval ;
    op.mat = mat ;
    op.loc1 = loc1;
end




