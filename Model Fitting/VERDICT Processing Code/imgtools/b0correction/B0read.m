function varargout = B0read( varargin )
% B0read
%
%  [vB0, mB0] = B0read( varargin )
%  [vB0, mB0, mask] = B0read( varargin )
%
% B0read( filenm )
% B0read( dinfo )
% B0read
%
% See also B0disp
%
thresh_mask = 10 ; % fraction of 95th percentile in data

if nargin == 0
    filenm = dselect('Name','Select B0 file') ;
elseif isstruct(varargin{1})
    filenm = varargin{1}.Filename ;
else
    filenm = varargin{1} ;
end

dB0 = datparse(filenm) ;

% Annoying changes in use of ImageType. For Philips Release 5.1.7, seems to be 20,
% whether EMR or Single? 

dB0uq = unique([dB0.itype]) ;
if ~isempty(intersect(dB0uq,20))
    itypeb0 = 20 ;
    itype_mod = 5 ;
else
    itypeb0 = 9 ;
end

[vB0, mB0] = d2mat(dB0,{'slice','itype'},'itype',itypeb0,'op','dv') ;

varargout{1} = vB0 ; varargout{2} = mB0 ;

if nargout == 3
    [vB0m, mB0m] = d2mat(dB0,{'slice','itype'},'itype',itype_mod,'op','fp') ;
    mask = ones(size(vB0m)) ;
    thresh = prctile(vB0m(:), 95) / thresh_mask ;
    loc = vB0m < thresh ;
    mask(loc) = 0 ;
    
    varargout{3} = mask ;
end

    
