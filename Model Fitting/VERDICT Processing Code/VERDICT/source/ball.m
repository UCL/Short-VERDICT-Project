function sig = ball(bval,dEES)
% BALL Simulate diffusion signal from a "ball" (unrestricted space)
%
%  sig = ball(bval,dEES)
%
% bval in s/mm^2, typical value 1500
% dEES in 1e-9, typical value 2
%
% Copyright 2022. David Atkinson
%
% See also stejskal sphereGPD astrosticks verdict_fit

arguments
    bval {mustBeNonnegative}
    dEES {mustBeNonnegative}
end

dEES_si  = 1e-9 * dEES; 
bval_si  = 1e6  * bval ;

sig = exp(-bval_si * dEES_si) ;