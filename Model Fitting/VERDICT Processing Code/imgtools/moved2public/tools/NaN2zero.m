function iout = NaN2zero(iin)
% NaN2zero  Sets any pixels with the value NaN to zero.
%
% function iout = NaN2zero(iin)
%
% David Atkinson, UMDS, June, 1998.
%
% @(#)NaN2zero.m	1.1 , created 07/01/98
%
% See also MASK2NAN SEGMENT EG

nans = isnan(iin) ;
setz = find(nans==1) ;
iout = iin ;
iout(setz) = zeros(size(setz)) ;
