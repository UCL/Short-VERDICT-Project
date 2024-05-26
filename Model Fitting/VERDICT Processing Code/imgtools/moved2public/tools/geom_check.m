function [slsep, rdc, cdc, sdc] = geom_check(geom)
% GEOM_CHECK Checks slices are parallel, calculates slice centre separation
% or thickness if one slice.
%
%  [slsep, rdc, cdc, sdc] = geom_check(geom)
%
%  geom is a structure with fields IOP, IPP, SliceThickness and possibly
%                                  MRAqType
%
% slsep - slice centre separation (may differ from Slice thickness).
%
% rdc, cdc, sdc - row, column and slice direction cosines
%
% David Atkinson  D.Atkinson@ucl.ac.uk
%
% See also DRESLICE  MPR_GEOM

THRESH = 0.001 ;

nsl = length(geom) ;

iop = geom(1).IOP ;
rdc = iop(1:3) ; % row direction cosine
cdc = iop(4:6) ;
sdc = cross(rdc,cdc) ; % slice direction cosine

slicepos = zeros([nsl 1]) ;

if ~isfield(geom,'MRAqType')
    is3D = false ;
    warning(['No MRAqType found, assume 2D'])
else
    is3D = true ;
end

for isl = 1:nsl
    % Check IOPs are parallel
    
    iop_this = geom(isl).IOP ;
    if ~isequal(iop, iop_this)
      warning(' Slices not parallel ')
    end
   
    slicepos(isl) = dot(sdc, geom(isl).IPP) ;
end

[srtslicepos, idx] = sort(slicepos) ;
if nsl == 1
    slsep = geom(1).SliceThickness ;
else
    slsep = srtslicepos(2)-srtslicepos(1) ;
    if is3D
        % slices should be along positive sdc
        sv = geom(2).IPP - geom(1).IPP ;
        % check parallel
        if norm(sv./norm(sv) - sdc) > 0.01
            warning(['3D slices not alignd with +ve slice direction'])
            disp(['sdc: ',num2str(sdc),' sv: ',num2str(s2s)])
        end
        if abs(norm(sv)-geom(1).SliceThickness) > 0.01
            warning(['Slice separation is not slice thickness in geom'])
            disp(['norm(s2s): ',num2str(norm(sv)),' SliceTickness: ',...
                geom(1).SliceThickness])
        end
    end
end

for isl = 3:nsl
    sdiff = srtslicepos(isl)-srtslicepos(isl-1) ;
    if abs(sdiff-slsep) > THRESH
        warning([' Slice separations not equal, ',...
            num2str(sdiff),'  slsep ',num2str(slsep)] )
    end
end
end