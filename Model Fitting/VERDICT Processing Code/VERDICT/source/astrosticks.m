function sig = astrosticks(bval,dVASC)
% ASTROSTICKS Astrosticks signal prediction
%
%  sig = astrosticks(bval,dVASC)
%
% dVASC in 1e-9 m^2/s typical value 8
% bval  in s/mm^2     typical value 1000
% 
% bval and dVASC must be scalar or have the same size
%
% Adapted from Astrosticks.java in Camino.
%   See equation A7 and associated references in 
%   E. Panagiotaki et al.  NeuroImage 59 (2012) 2241–2254
% Note there are alternative ways to perform this computation, for example
% AstroSticks_GEN in the MISST toolbox that explicitly sums over a
% distribution of fibres.
%
% Copyright 2022. David Atkinson.
%
% See also sphereGPD ball verdict_fit

arguments
    bval {mustBeNonnegative}
    dVASC {mustBeNonnegative}
end

dVASC_si = 1e-9 * dVASC; 
bval_si  = 1e6  * bval ;

% Gsqrtpar here is G * sqrt(-lpar) from CAMINO. lperp assumed to be 0.

if bval_si == 0
    sig = 1 ;
else
    Gsqrtpar = sqrt( bval_si .* dVASC_si ) ;

    sig = sqrt(pi) / (2 .* Gsqrtpar) .* erf(Gsqrtpar) ;
end