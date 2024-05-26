function [outputCoord,varargout] = coord2D3D(inputCoord,geom, options)
% COORD2D3D Change between 2D image ROI coordinates and 3D Patient Coordinates
% Can transform 2D to 3D or 3D to 2D.
%  
% [coord2D, distances] = coord2D3D(coord3D, geom) 
% [coord2D, distances] = coord2D3D(coord3D, geom, Name, Value, ...) 
% [coord3D ] = coord2D3D(coord2D, geom)  (For coord2D in intrinsic coords)
%
% coord2D is [N 2] in [x y] (NOT row column)
% coord3D is [N 3] in [L P S]
%
% Name, Value
% 'type2D' can be  '2Dintrinsic' (default) or '2Dmm'
%  '2Dintrinsic' corresponds to intrinsic coords from e.g. imshow, where top
%  left center is (1,1) and pixels are unit size.
%  '2Dmm' corresponds to coordinates with non-default XData and YData (as 
%   used in sviewer) where the in-plane origin is the closest point to the
%   DICOM Patient origin and pixel units are mm.
%
% 'threshParallel'  default 0.8 
%
% Examples
%
% Convert 3D coordinates from a DICOM RTSTRUCT in patient LPH system to the
% MATLAB default 2D image coordinate system for an image that is slice
% number islice with associated geom structure. 
%  coord2D = coord2D3D(coordsROI3D, geom(islice))
%
% Copyright 2022, David Atkinson
%
% See also display_rtstruct
%

arguments
    inputCoord {mustBeNumeric}
    geom 
    
    % ROI plane and image plane normal dot products 
    % should exceed this (otherwise very not parallel)
    options.threshParallel (1,1) {mustBeFinite} = 0.8 
    options.type2D {mustBeNonzeroLengthText} = '2Dintrinsic'
end

if ~isstruct(geom) || ~isfield(geom,'IOP')
    error('MATLAB:coord2D3D:geomInput', ...
        'geom input needs to be a geom structure including field IOP')
end


cinputdim = size(inputCoord,2) ; % 2D or 3D for input coords
np = size(inputCoord,1) ; % number of points in ROI

varargout = {} ;

% Need to use consistent row or column vector manipulation
% Coords are Nx2 or Nx3 so use row vectors
if iscolumn(geom.IPP)
    geom.IPP = geom.IPP' ;
end

if iscolumn(geom.IOP)
    geom.IOP = geom.IOP' ;
end

switch cinputdim
    case 2  % Input is 2Dintrinsic or 2Dmm
            % Output is 3D (in LPH / mm)
        outputCoord = zeros([np 3]) ;

        switch options.type2D
            case '2Dmm'
                pixel_scale_HW = [1 1] ; 
                tlc_xy = [geom.XData(1) geom.YData(1)] ;
            case '2Dintrinsic'
                pixel_scale_HW = geom.PixelSpacing_HW ;
                tlc_xy = [1 1] ;
        end

        for ip = 1:np
            outputCoord(ip,:) = geom.IPP + ...
                (inputCoord(ip,1)-tlc_xy(1)) * pixel_scale_HW(2) * geom.IOP(1:3) + ...
                (inputCoord(ip,2)-tlc_xy(2)) * pixel_scale_HW(1) * geom.IOP(4:6) ;
        end


    case 3
        % Check input points are coplanar
        [ncp, ~] = plane_fit(inputCoord) ;
        if iscolumn (ncp)
            ncp = ncp' ;
        end

          % Check ROI plane is roughly parallel to image slice
        nsp = cross(geom.IOP(1:3),geom.IOP(4:6)) ; % normal to slice

        projn = abs(dot(nsp,ncp)) ;
        if projn < options.threshParallel
            warning('MATLAB:coord2D3D:ROIandImagePlanesNotParallel', ...
                'ROI and Image planes not parallel')
        end

        % Calculate 2D image points as closest point of 3D ROI on image
        % plane

        % See https://en.wikipedia.org/wiki/Line–plane_intersection

        % Vector equation of line from ROI point in a) direction of normal
        % to ROI, and, b) direction of normal to image plane.

        % Plane of interest is image plane with normal nsp 

        outputCoord = zeros([np 2]) ;
        distpROI2plane = zeros([1 np]) ;

        % Code below is designed to cope with a contour that is not exactly
        % in the plan of the image slice. Points on the ROI are projected
        % onto the image plane along the direction nsp of the slice normal.
        % Only tested for contour planes coincident with image plane.
        for ip = 1:np
            ll = nsp ; % vector in direction of line (here nsp, normal to image)
                       % would be replaced with ncp for contour plane
            p3 = inputCoord(ip,:) ; % point on line
            P0 = geom.IPP ; % point in plane
            d = dot((P0 - p3), nsp) / dot( ll, nsp) ;

            pintersect = p3 + (ll*d) ;

            % convert pintersect (3D position of point in image plane, to
            % 2D)

            % xmm and ymm are distances from point in plane to IPP
            xmm = dot((pintersect-geom.IPP), geom.IOP(1:3)) ;
            ymm = dot((pintersect-geom.IPP), geom.IOP(4:6)) ;
            
            % convert to intrinsic for use with imshow
            x = 1 + xmm/geom.PixelSpacing_HW(2) ;
            y = 1 + ymm/geom.PixelSpacing_HW(1) ;

            switch options.type2D
                case '2Dintrinsic'
                    outputCoord(ip,:) = [ x, y] ;
                case '2Dmm'
                    outputCoord(ip,:) = [ xmm+geom.XData(1) ymm+geom.YData(1) ] ;
                otherwise
                    error('MATLAB:coord2D3D:unknownType2D',...
                        ['Unknown type2D: ',options.type2D])
            end

            distpROI2plane(ip) = d ;
        end

        varargout{1} = distpROI2plane ;

    otherwise
        error('MATLAB:coord2D3D:inputCoordShape','Input needs to be Nx2 or Nx3')
end

end



