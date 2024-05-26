function dout = addimnoise(din, snr, signal)
% ADDIMNOISE Adds complex noise in image domain.
% dout = addimnoise(din, snr, signal)
%
% dout = addimnoise(din, snr) 
%  Signal is estimated based on average image value 
%  (image is taken to be all values over a threshold
%   based on the maximum).
%
% snr here is the ratio of the signal to the variance of
%     the complex noise (prior to taking image magnitude).
%
% dout is complex
%
% Example:
%  imnoisy = abs(addimnoise(imin, 10)) ;
%
% variance  = signal / snr
% dout = din + rand(0 mean, variance 1)*data_max/var_denom +i*...
%
% D.Atkinson@ucl.ac.uk   
% $Id: addimnoise.m 180 2007-11-29 18:10:24Z ucacdat $
% See also addnoise

if nargin < 3
    % estimate signal if not input
    adin = abs(din(:)) ;
    
    dmax = max(adin) ;
    thresh = dmax / 15 ;
    loc = find(adin > thresh) ;
    signal = sum(adin(loc)) / length(loc) ;
end
    
scale = signal / snr ; 

dout = din + randn(size(din))*scale + 1i*randn(size(din))*scale ;

