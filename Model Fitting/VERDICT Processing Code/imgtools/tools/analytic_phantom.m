function varargout = analytic_phantom(geom1, arg2)
% 
% Analytic phantom
%
% [vol, geom] = analytic_phantom(geom1, nslice) ; 
% pv          = analytic_phantom(geom1, icoords) ;
%
% icoords [nc 3] stacked instrinsic coords in x,y,z with top 
% left centre at [1,1,1]. Output pv is pixel values.
%
% See also imzoneplate

if length(geom1) > 1
    error('geom1 must refer to a single slice (first in volume or ref for coords)')
end

% Object limits in LPH
Ll = 9.5 ; Lu = 90.5 ;  % |   |   |xxx    |
                        %   8   9   10
Pl = 19.5 ; Pu = 70.5 ;
Hl = 2.5  ; Hu = 25.5  ;


if length(arg2) ==1
    coordin = false ;
else
    coordin = true ;
    icoords = arg2 ;
end

if coordin == false 
    nx = geom1.Width ;
    ny = geom1.Height  ;
    nz = arg2 ;
    
    [ix, iy, iz] = meshgrid(1:nx, 1:ny, 1:nz) ;
    icoords = cat(2,ix(:), iy(:), iz(:)) ;
    
    for iz = 1:nz
        geom(iz).Width = geom1.Width ;
        geom(iz).Height = geom1.Height ;
        geom(iz).PixelSpacing_HW = geom1.PixelSpacing_HW ;
        geom(iz).IOP = geom1.IOP ;
        geom(iz).SliceThickness = geom1.SliceThickness ;
        
        [ipp_lph] = dicom2intrinsic(geom1, 'output', 'stackedLPH', ...
                                                   'coords', [1 1 iz]) ;
                                               
        geom(iz).IPP = ipp_lph ;
    end
        
end
  
[coords_lph] = dicom2intrinsic(geom1, 'output', 'stackedLPH', ...
                                                   'coords', icoords) ;
L = coords_lph(:,1) ;
P = coords_lph(:,2) ;
H = coords_lph(:,3) ;
  
nc = size(icoords,1) ; 
out = zeros([nc 1]) ;

for ic = 1:nc
    if L(ic)<Ll || L(ic)>Lu || P(ic)<Pl || P(ic)>Pu || H(ic)<Hl || H(ic)>Hu
        out(ic) = 0 ;
    else
        
        
        % Code below is adapted from imzoneplate
        %   Copyright 2012 The MathWorks, Inc.
        %   Steven L. Eddins
        r = hypot(L(ic),P(ic));
        km = 0.7*pi;
        rm = 64; % assumes a 128 image
        w = rm/10;
        term1 = sin( (km * r.^2) / (2 * rm) );
        term2 = 0.5*tanh((rm - r)/w) + 0.5;
        g = term1 .* term2;
        
        out(ic) = (g + 1)/2;
        % out(ic) = 1 ;
    end
end

if coordin == false
    out = reshape(out,[ny nx nz]) ;
end

varargout{1} = out ;

if coordin == false 
    varargout{2} = geom ;
end
