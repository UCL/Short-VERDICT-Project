function [ADC] = dwi2ADC(DWI, B0, bv)
% DWI2ADC Three orthogonal DWI to ADC
% [ADC] = dwi2ADC(DWI, B0, bv)
%
%
% D.Atkinson@ucl.ac.uk
%
% See also calcADC

szDWI = size(DWI) ;

% check sizes
if size(B0,3)>1
    is3D=true;
else
    is3D=false;
end

if szDWI(1) ~= size(B0,1) | szDWI(2) ~= size(B0,2)
    error(['DWI and B0 must have same numbers of rows and columns'])
end

nd = szDWI(end) ;
if nd~=3
    warning(['Expected 3 orthogonal diffusion directions, not: ',num2str(nd)])
end

if is3D
    if szDWI(3) ~= size(B0,3)
        error(['DWI and B0 must have same number of slices'])
    end
    
    ddim = 4 ;
    B0 = repmat(B0,[1 1 1 nd]) ;
else
    B0 = repmat(B0,[1 1 nd]) ;
    ddim = 3;
end


log_ratio = log(B0./DWI );

ADCd = log_ratio / bv ;
loc = ~isfinite(ADCd) ;
ADCd(loc) = 0 ;

ADC = sum(ADCd,ddim)/nd ;


