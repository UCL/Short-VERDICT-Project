function c = coil_sensitivity(a,R,Z)
% COIL_SENSITIVITY
%
%  c = coil_sensitivity(a,R,Z)
%
%  a the coil radius (scalar)
%  R distances to coil axis
%  Z distances to coil plane
%
% R and Z can be matrices but must be the same size.
%
% From Wang MRM 44:495 eqn 10 (taken from Landau and Lifshitz)
%
% @(#)coil_sensitivity.m	1.1 , created 03/08/02  at KCL
%
% D.Atkinson@ucl.ac.uk
% $Id: coil_sensitivity.m 153 2007-06-25 16:32:52Z ucacdat $

R = abs(R) ;
Z = abs(Z) ;

if ~isequal(size(R), size(Z)) 
  error(['R and Z must be the same size'])
end

aRZ = (a+R).^2 + Z.^2 ;
K2 = 4*a*R ./( aRZ ) ;

[K,E] = ellipke(K2) ;

c = 2.*(K + (a.^2 - R.^2 - Z.^2)./((a-R).^2 + Z.^2) .*E ) ./sqrt(aRZ) ;


