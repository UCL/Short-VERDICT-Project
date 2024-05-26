function [DC, kdc] = dc3D(k_in)
% dc3D Finds the 3D DC co-ordinate in k-space (Co-ord with maximum absolute
%                                                  value)
% [DC, kdc] = dc3D(k_in)
% 
% DC  is a row vector with 3 elements WITH Y,X,Z ORDERING
% kdc is the k-space at this point.
%
% dc3D(k_in) displays the values on the screen
%
% David Atkinson
% The Guy's, King's and St. Thomas' School of Medicine, 1998.
% @(#)dc3D.m	1.2 , created 11/16/98
%

Y=1; X=2; Z=3;

Ak = abs(k_in) ;

[DC_ky, DC_kx, DC_kz] = ind2sub(size(k_in), find( Ak == max(max(max(Ak)))) );

if length(DC_ky) ~= 1
  warning([ ' Found ',num2str(length(DC_ky)), ...
	' points in abs(k_space) with the same maximum value.'])
  
  % If grid point max is in the maxima, return that.
  
  dcgp = DCgp(k_in) ;
  
  locy = find(DC_ky == dcgp(Y)) ; 
  locx = find(DC_kx == dcgp(X)) ;
  locz = find(DC_kz == dcgp(Z)) ;
  
  intxy = intersect(locx, locy) ;
  intxyz = intersect(intxy, locz) ;
  
  if length(intxyz) == 1
    disp([ ' Returning the grid point DC.'])
    DC_ky = DC_ky(intxyz) ;
    DC_kx = DC_kx(intxyz) ;
    DC_kz = DC_kz(intxyz) ;
  else
    disp([ ' Returning the first max value.'])
    DC_ky = DC_ky(1); 
    DC_kx = DC_kx(1); 
    DC_kz = DC_kz(1) ;
  end

end

nnans = sum(sum(sum(isnan(Ak)))) ;
  
if nnans ~= 0
  warning([ '!!! You have ',num2str(nnans),' NaNs in your data !!!!!!' ])
end

if nargout == 0
  disp([ 'DC_kx = ',num2str(DC_kx), ...
	 '  DC_ky = ',num2str(DC_ky), ...
	 '  DC_kz = ',num2str(DC_kz) ])
else
   DC = [DC_ky, DC_kx, DC_kz] ;
   kdc = k_in(DC_ky,DC_kx,DC_kz) ;
   return
end
 
 

