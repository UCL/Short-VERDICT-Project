function [pdata, rs, ri] = dscale(indata,type,drange, method)
% DSCALE Data scale prior to output
%
% [pdata, rs, ri] = dscale(data,type,drange, method)
%
% data - data to be rescaled
% type - output type e.g. 'uint16'
% drange [min_data  max_data] (e.g. data could be one slice and drange for
%     all volume)
% method  'full'  uses full range of type
%         'pow10' rescales only by a power of 10, intercept 0. 
%         'none' rs = 1, ri = 0. Will clip outside of type range.
%
% pdata  output data 
% rs RescaleSlope
% ri RescaleIntercept
%
% David Atkinson  D.Atkinson@ucl.ac.uk
%
% See also writeDicom
%

switch type
    case 'uint16'
        vmax = double(intmax(type)) ;
        vmin = double(intmin(type)) ;
    otherwise
        error(['Type not implemented'])
end
 
switch method
    case 'full'
        ri = drange(1) ;
        rs = (drange(2) - drange(1)) / (vmax-vmin)  ;
    case 'pow10'
        ri = 0 ;
        orig_rs = (drange(2) - drange(1))/ (vmax-vmin)  ;
        rs = 10^(floor(log10(orig_rs))) ;   
    case 'none'
        ri = 0 ;
        rs = 1 ;
    otherwise
        error(['Not implemented'])
end

data = (indata - ri ) / rs ;

switch type
    case 'uint16'
        pdata = uint16(data) ; % rounds and clips if required
end