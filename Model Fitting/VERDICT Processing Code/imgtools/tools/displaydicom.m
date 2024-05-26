function displaydicom(varargin)
% DISPLAYDICOM Displays DICOM files in 3D
%
%  displaydicom('dinfo', dinfo)
%
% Example
%   dinfo = dmfparse(dselect) ;
%   displaydicom('dinfo', dinfo)
%
% D.Atkinson@ucl.ac.uk
%


p = inputParser ;
addParameter(p,'dinfo','') ;

parse(p,varargin{:}) ;
dinfo = p.Results.dinfo ; 


ndf = length(dinfo) ; % nunber of DICOM frames

for idf = 1: ndf
    img = dicomread(dinfo(idf).Filename,'frames',idf) ;
    
    nrow = double(dinfo(idf).Height);
    ncol = double(dinfo(idf).Width) ;
    ps = dinfo(idf).PixelSpacing ;
    iop = dinfo(idf).ImageOrientationPatient ;
    ipp = dinfo(idf).ImagePositionPatient ;
    
    
    % form matrices X,Y and Z to contain image
    % vertex coordinates
    X = zeros(nrow+1, ncol+1) ;
    Y = zeros(nrow+1, ncol+1) ;
    Z = zeros(nrow+1, ncol+1) ;
    
    % 3D coord of top left vertex.
    %
    %  tlv
    %  +-----+-----+-----+
    %  |     |     |     |
    %  |  x  |     |     |
    %  |     |     |     |
    %  +-----+-----+-----+
    %  |     |     |     |
    %  |     |     |     |
    %
    % x is centre of top left coord, this is
    % ipp in the patient coordinate system.
    % To compute tlv (top left vertex), go
    % back half a pixel and up half a pixel.
    % Scale for the pixel dimensions
    %
    % Take care over row/column spacing, directions etc
    tlv = ipp + -0.5*(iop(1:3)*ps(2) + iop(4:6)*ps(1))  ;
    
    % loop over each vertex
    for ir = 1:nrow+1
        for ic = 1:ncol+1
            % vertex coordinate is just the coord of the
            % top left vertex plus a vector down the rows
            % plus a vector along the columns.
            coordv = tlv + (ir-1)*iop(4:6)*ps(1) + ...
                (ic-1)*iop(1:3)*ps(2) ;
            
            X(ir,ic) = coordv(1) ; % separate the x component into the matrix X,
            Y(ir,ic) = coordv(2) ; % the y component into Y etc
            Z(ir,ic) = coordv(3) ;
        end
    end
    
    % Surf plot
    surf(X,Y,Z,img,'EdgeColor','None')
    
    hold on
    axis equal
    axis vis3d
    grid
    view(21,-50)
    xlabel('x left') ; ylabel('y posterior') ; zlabel('z head')
    colormap gray
    
end


