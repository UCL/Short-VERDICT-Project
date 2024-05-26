function geom = dgeomextract(dinfo)
% DGEOMEXTRACT Extract geometry information from dinfo structure
%
% geom = dgeomextract(dinfo)
%
% dinfo is a structure read in using datparse, dmfparse, parparse, 
%  dparse or xmlparse
%  Angulations and offcentres will be converted to IOP and IPP
%
% geom is a structure with fields:
%  IPP, IOP, Width, Height, PixelSpacing_HW, SliceThickness, XData, YData, 
%  MRAqType, source, R2D, D
%
% R2D is a 2D apatial referencing object for MATLAB registration
% D is the distance of the spatial referencing plane from the DICOM origin.
%
% 
% David Atkinson   D.Atkinson@ucl.ac.uk
%
% See also DGEOM DATPARSE GEOM2SRO
%


sls = [dinfo.sl] ;

[usl, loc] = unique(sls) ;

nsl = length(usl) ;

disp([num2str(nsl),' unique slices.'])

geom = struct('IPP',num2cell(zeros([nsl 3]),2), ...
    'IOP',num2cell(zeros([nsl 6]),2), ...
    'Height',num2cell(zeros([nsl 1])), ...
    'Width',num2cell(zeros([nsl 1])), ...
    'PixelSpacing_HW',num2cell(zeros([nsl 2]),2), ...
    'SliceThickness',num2cell(zeros([nsl 1])), ...
    'XData',num2cell(zeros([nsl 2]),2), ...
    'YData',num2cell(zeros([nsl 2]),2), ...
    'MRAqType', num2cell(repmat('??',[nsl 1]),2), ...
    'source',{'Data source'}) ;

 
FrefUID = {dinfo(loc(:)).FrameOfReferenceUID} ;
uq_Fref = unique(FrefUID) ;

if length(uq_Fref) > 1
    warning(['FrameOfReferenceUIDs differ: possible misalignment. Non unique:',num2str(length(uq_Fref))])
end

% Check MRAcquisitionType (required for dreslice to choose kernel).
force2D = false ;
if ~isfield(dinfo,'MRAcquisitionType')
    warning(['No MRAcquisitionType (2D, 3D) set in input'])        
else
    switch dinfo(loc(1)).MRAcquisitionType
        case '3D'
            if nsl == 1
                warning(['3D acquisition, but only one slice.'])
                disp(['Setting MRAcquisitionType to 2D'])
                force2D = true ;
            end
   
        case '2D'
        otherwise
            warning(['Unknown MRAcquisitionType (expect 2D or 3D)'])
    end
end
        
for isl = 1:nsl
    PS_HW = dinfo(loc(isl)).PixelSpacing ;
    Width = double(dinfo(loc(isl)).Width) ;
    Height = double(dinfo(loc(isl)).Height) ;
    SliceThickness = dinfo(loc(isl)).SliceThickness ;
    if ~isfield(dinfo,'ImagePositionPatient')
        geom(isl).source = 'PARREC' ;
        ori = dinfo(loc(isl)).SliceOrientation ;
        offc = dinfo(loc(isl)).Offcentre_lph ;
        ang = dinfo(loc(isl)).ImageAngulation_lph ;
        
        [iop, ipp] = dgeom(offc, ang, ori, PS_HW , Width, Height);
    else
        geom(isl).source = 'DICOM' ;
        ipp = dinfo(loc(isl)).ImagePositionPatient ;
        iop = dinfo(loc(isl)).ImageOrientationPatient ;
    end
    geom(isl).IPP = ipp ;
    geom(isl).IOP = iop ;
    geom(isl).Width = Width;
    geom(isl).Height = Height ;
    geom(isl).SliceThickness = SliceThickness ;
    geom(isl).PixelSpacing_HW = PS_HW ;
    
    % set XData and YData consistently based on origin (for viewing)
    plane = set_plane('pvv',ipp,iop(1:3),iop(4:6)) ;
    d = point_plane([0 0 0],plane) ; %Patient coord origin to plane distance
    oip = d*plane.normal ; %closest point in plane of origin 
    oip2ipp = ipp - oip ;
    XData(1) = dot(oip2ipp,iop(1:3)) ;
    YData(1) = dot(oip2ipp,iop(4:6)) ;
    XData(2) = XData(1) + PS_HW(2)*(Width-1) ;
    YData(2) = YData(1) + PS_HW(1)*(Height-1) ;
    
    % Note above can be done as oip = -D.*n where 
    % n=cross(IOP(1:3),IOP(4:6))
    % d = -dot(n,IPP) ; D = d/norm(n,2) ;
    
    geom(isl).XData = XData ;
    geom(isl).YData = YData ;
    
    if exist('imref2d','file') % spatial referencing not in old MATLAB
        [R2D,D] = geom2sro(geom(isl)) ;
        geom(isl).R2D = R2D;
        geom(isl).D = D ;
    end
    
    if isfield(dinfo,'MRAcquisitionType')
        if force2D
            geom(isl).MRAqType = '2D' ;
        else
            geom(isl).MRAqType = dinfo(loc(isl)).MRAcquisitionType ;
        end
    end
    
end


end




