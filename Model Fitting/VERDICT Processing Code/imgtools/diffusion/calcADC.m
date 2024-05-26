function [ADC, S0] = calcADC(volb, bvVec, bvUse)
% calcADC Calculate ADC by linear fitting to log of data (multiple b
% values)
%
% [ADC, S0] = calcADC(volb, bvVec)
% [ADC, S0] = calcADC(volb, bvVec, bvUse)
%
% volb [ny nx nz nbv]  or  [ny nx nbv]
% bvVec should have length nbv
% bvUse defaults to bvVec, otherwise only the bvalues present in bvUse will
% be included in the ADC calculation
%
% Solves the following for S0 and ADC using a least squares fit 
% to the log of the data:
%   S=S0 exp(-ADC.b)
% i.e. 
% ln(S) = ln(S0) - ADC.b
% ln(S) = ADC.(-b) + ln(S0)
% Y = mX + c
% Y = [X 1] [m c]'
%
% 
% Examples
% dinfo = datparse ;
% [volb, matp] = d2mat(dinfo,{'slice','bv'},'op','fp') ;
% [ADC, S0] = calcADC(volb, matp.bvVec) ;
%  displayADC(ADC*1e6, 2000)
%
% dinfo = dfparse(ffn) ;
% [vb0, mb0] = d2mat(dinfo,{'slice','bv'},'bv',0,'op','fp%) ;
% bvs = unique([dinfo.DiffusionBValue]) ;
% [vbiso, mbiso] = d2mat(dinfo,{'slice','bv','ddty'},'ddty',2,'bv',bvs(2:end),'op','fp%) ;
% volb = cat(4, vb0, vbiso);
% [ADC, S0] = calcADC(volb, bvs) ;
%
%
%
% David Atkinson
%
% See also dwi2ADC

% Reshape so that single slice is still 4D (inc bvaues)
ndv = ndims(volb) ;
if ndv==3
    volb = reshape(volb,[size(volb,1) size(volb,2) 1 ndv]) ;
end

volb(~isfinite(volb)) = 0 ;
loc = volb < 0 ;

if ~isempty(loc)
    warning('Negative signal in input to calcADC')
    volb(loc) = 0 ;
end


% Restrict bvalues if requested.
if nargin == 3
   [bvIntersect, ia] = intersect(bvVec,bvUse) ;
   if length(bvUse)~=length(bvIntersect) 
       warning(['Not all b-values in bvUse are in data'])
   end
   
   bvVec = bvVec(ia) ;
   volb = volb(:,:,:,ia) ;
end
    
[ny nx nz nbv] = size(volb) ;
if nbv ~= length(bvVec) 
    error(['Last dimension of volb must be same as number of bvalues'])
end

A = cat(2, -bvVec(:), ones([length(bvVec) 1])) ;

B = log(volb) ;
B = reshape(B,[ny*nx*nz nbv]) ;

Btest = sum(B,2) ;
[row] = find(~isfinite(Btest)) ;
B(row, :) = 0 ;

B = B.' ;

X = A\B ;

ADC = X(1,:) ;
S0 = X(2,:) ;

ADC(row) = 0 ;
S0(row) = 0 ;

ADC = reshape(ADC,[ny nx nz]) ;
S0 = exp(reshape(S0,[ny nx nz])) ;


