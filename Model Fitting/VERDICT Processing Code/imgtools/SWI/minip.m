function [mslide, mslab] = minip(data, nsgp)
% MINIP Minimum Intensity Projection
% 
% Typically called after swi2D
% 
%  [mslide, mslab] = minip(data, nsgp)
%
% David Atkinson  D.Atkinson@ucl.ac.uk
% See also SWI2D 
%

[ny nx nz] = size(data) ;

if rem(nsgp,2)==0
    warning(['Even number of slices in group.'])
    return
else
    nmid = (nsgp+1)/2 ;
end

% Sliding mIP calculation
mslide = zeros([ny nx nz]) ;
for islice = nmid:nz-nmid+1
    mslide(:,:,islice) = min(data(:,:,islice-nmid+1:islice+nmid-1 ),[],3) ;
end

% mIP per group of slices
slabs = 1:nsgp:nz ;
nslab = length(slabs) - 1 ;

mslab = zeros([ny nx nslab]) ;
for islab = 1:nslab
    mslab(:,:,islab) = min(data(:,:,slabs(islab):slabs(islab+1)-1), [],3) ;
end
