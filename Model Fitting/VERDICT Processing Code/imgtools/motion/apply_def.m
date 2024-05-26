function [ Iout ] = apply_def(Ux, Uy, Iin, interpol  )
%APPLY_DEF Apply deformation field
%   
% [ Iout ] = apply_def(Ux, Uy, Iin  )
%    W (lanczos-sinc kernel size) defaults to 6
% [ Iout ] = apply_def(Ux, Uy, Iin, W  )
% [ Iout ] = apply_def(Ux, Uy, Iin, Interpolator  )
%
% David Atkinson, adapted from code by Freddy Odille
% UCL, 2012
%
% See also OPT_FLOW  GEN_OPTICAL_FLOW_2D

if nargin < 4
    interpol = 6 ;
end

if isstruct(interpol)
    LookupTable = interpol.LookupTable ;
    nbins = interpol.nbins;
    W = interpol.W ;
else
    W = interpol ;
    nbins = 2^16 ;
    xList = -W/2 : W/(nbins-1) : W/2 ;
    if(W==2)
      LookupTable = 1 - abs(xList);
    else
      LookupTable = sinc(xList) .* sinc( xList / (W/2) ); % lanczos-sinc
    end
end

[ny, nx, nt] = size(Iin) ;
Iout = zeros([ny nx nt]) ;

for t=1:nt
  U_t = zeros([ny nx 1 2]) ;
  U_t(:,:,:,1)= Uy(:,:,t) ;
  U_t(:,:,:,2)= Ux(:,:,t) ;
  
  Iout(:,:,t) = interp_mex( double(Iin(:,:,t)), double(U_t), ...
                    LookupTable, nbins, W, 0 );
end

