function [T2, S0] = calcT2(data, TEvec)

arguments
    data % stacked array of images from different echos [..., necho]
    TEvec % vector of echo times (necho length)
    
end

[ny, nx, nz, necho] = size(data) ;
if necho ~= length(TEvec) 
    error(['Last dimension of data must be same as number of TEs'])
end

A = cat(2, -TEvec(:), ones([length(TEvec) 1])) ;

B = log(data) ;
B = reshape(B,[ny*nx*nz necho]) ;

Btest = sum(B,2) ;
[row] = find(~isfinite(Btest)) ;
B(row, :) = 0 ;

B = B.' ;

X = A\B ;

T2 = 1./X(1,:) ;
S0 = X(2,:) ;

% Remove meaninless measurements
T2(T2<0) = 0;
T2(S0<0) = 0;
S0(T2<0) = 0;
S0(S0<0) = 0;
T2(row) = 0 ;
S0(row) = 0 ;

T2 = reshape(T2,[ny nx nz]) ;
S0 = exp(reshape(S0,[ny nx nz])) ;


end