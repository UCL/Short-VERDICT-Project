function [SM] = gen_sincmat(reg, ireg)
% GEN_SINCMAT  Generate sinc matrix for k-space interpolation in fwd SENSE
%
%  SM = gen_sincmat(reg, ireg) 
% reg is a vector of regularly spaced coordinates with step size 1
% ireg are the coordinates of the points to be interpolated.
% 
% SM  [ nireg nreg]
%
% D.Atkinson@ucl.ac.uk
%

nireg = length(ireg) ;
nreg =  length(reg)  ;

reg = reg(:)' ; % row vector
ireg = ireg(:) ; % column vector

coordmat = reg - ireg ; % Using MATLABs expansion properties

SM = sinc(coordmat) ;



