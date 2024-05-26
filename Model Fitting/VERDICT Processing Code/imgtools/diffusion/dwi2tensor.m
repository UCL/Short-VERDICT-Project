function [D,S0] = dwi2tensor(DWI, varargin)
%DWI2TENSOR DWI to tensors using B matrix or gradients 
% [D,S0] = dwi2tensor(DWI, Bvec)
% [D,S0] = dwi2tensor(DWI, grad, bv) 
%
% D      = dwi2tensor(DWI, grad, bv, B0)
%
% Method 1:  Solves  S=S0 exp(-B:D)
% The low b-value images (including B0) are included with the DWI images 
% and a B0 image is returned as part of the fitting. 
% The gradients can be specified in x, y and z, or, the six 
% elements of the B matrix.
%
% Method 2: Solves  S=B0 exp( -bv  g^T D g)
% B0  is supplied separately and is not part of the fitting.
% The normalised x,y,z gradients should be supplied.
%
% DWI [ ny nx .... ngrad]
% grad [ngrad 3]  gradients (should be normalised when bv is supplied)
% Bvec [ngrad 6]  xx yy zz xy xz yz elements of B matrix
%
% D  [ny nx ... 3 3]
% S0 [ny nx ...]
% 
% See Kingsley Concepts MR 28A p 155 -  (Part III). Part of code adapted
% from Philip Batchelor
%
% David Atkinson  D.Atkinson@ucl.ac.uk
% $Id: dwi2tensor.m 223 2009-02-04 17:23:17Z ucacdat $
%
% See also dwi2tensorb

if nargin > 3
    b0out = 0;
    B0 = varargin{3} ;
    B0 = abs(B0) ;
else
    b0out = 1 ;
end

if nargin > 2
    bv = varargin{2} ;
else
    bv = 1 ;
end

DWI = abs(DWI) ;
szDWI = size(DWI) ;
npixels = prod(szDWI(1:end-1)) ;
ngrad = szDWI(end) ;

Bvec = varargin{1} ;
if size(Bvec,1) ~= ngrad
    error('dwi2tensor: no. of gradient directions and data not consistent')
end

if size(Bvec,2) ==3 % (x y z gradients, convert to xx yy zz xy xz yz)
    % check input gradients were normalised for second method
    grad = Bvec ;
    
    %DA method
    nrgrad = sqrt(dot(grad, grad, 2)) ;
    loc = find(abs(nrgrad-1) > 0.001) ;
    if ~isempty(loc) && b0out==0
        warning(['dwi2tensor: Normalising input gradients'])
        grad = grad./nrgrad(:,[1,1,1]) ;
    end
    
    if length(bv)~=1
        bv = repmat(bv(:),[1 6]) ;
    end
    
    Bvec = bv .* [grad.*grad  2*grad(:,[1,1,2]).*grad(:,[2,3,3])] ;
    
elseif size(Bvec,2) ~=6
    error(['dwi2tensor: gradient data must have 3 or 6 elements in each row'])
end   

DWI = reshape(DWI(:), [npixels ngrad]) ;
DWI = permute(DWI,[2 1]) ;


switch b0out
    case 1
        A = [-Bvec repmat(1,[ngrad 1])] ; 
        
        ws2 = warning('off','MATLAB:log:logOfZero') ;
        b = log(DWI) ;
        warning(ws2.state, ws2.identifier) ;

    case 0
        A = Bvec ;
        
        B0 = repmat(transpose(B0(:)),[ngrad 1]) ;
        
        ws1 = warning('off','MATLAB:divideByZero') ;
        ws2 = warning('off','MATLAB:log:logOfZero') ;
        b = -log(DWI./B0) ;
        warning(ws1.state, ws1.identifier) ; % return warning state to previous
        warning(ws2.state, ws2.identifier) ;
end

matD = A\b;

matD = permute(matD,[2 1]) ;

D = zeros([npixels 3 3]) ;

D(:,1,1) = matD(:,1) ; 
D(:,2,2) = matD(:,2) ;
D(:,3,3) = matD(:,3) ;
D(:,1,2) = matD(:,4) ;
D(:,2,1) = matD(:,4) ;
D(:,1,3) = matD(:,5) ;
D(:,3,1) = matD(:,5) ;
D(:,3,2) = matD(:,6) ;
D(:,2,3) = matD(:,6) ;

D = reshape(D,[szDWI(1:end-1) 3 3]) ;
D(~isfinite(D))=0;


switch b0out
  case 1
    S0 = exp(matD(:,7)) ;
    S0 = reshape(S0, [szDWI(1:end-1) 1]) ;
    S0(isnan(S0))=0 ;
end

