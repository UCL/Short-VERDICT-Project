function lwfn = vislwf(lwf,inp, outp)
% VISLWF Multi-linear image intensity transform for visualisation of LWF.
%  Allows for intensity 'amplification' or 'supression' for various lwf
%  ranges - intended to focus on the interesting region.
%  Hasn't proved to be very useful.
%
%  lwfn = vislwf(lwf,inp, outp)
%
% lwf - input luminal water fraction, values outside [0 1] will be clipped
% inp - 2-element vector with input image intensities for linear sections
% outp - 2-element vectr of corresponding output inensities
%
%
% David Atkinson  D.Atkinson@ucl.ac.uk
%

if max(inp) > 1 || max(outp)>1 || min(inp)<0 || min(outp)<0 
    error(['inp and outp must be in range [0 1]'])
end

lwfg = mat2gray(lwf,[0 1]) ;

%   1                           *
%                           *
%                      *
% outp(2)            *
%                   *
%                  *
%                 *
% outp(1)        *
%          *   *
%          0   inp(1)   inp(2)   1
%
%  Region    A       B      C

% find input regions
%

locA = lwfg<inp(1) ;
locB = lwfg>=inp(1) & lwfg<inp(2) ;
locC = lwfg>=inp(2) ;

lwfn = zeros(size(lwfg)) ;

lwfn(locA) = outp(1)*lwfg(locA)/inp(1) ;

lwfn(locB) = outp(1) + (lwfg(locB)-inp(1))/(inp(2)-inp(1))*(outp(2)-outp(1)) ;

lwfn(locC) = outp(2) + (lwfg(locC)-inp(2))/(1-inp(2))*(1-outp(2)) ;
