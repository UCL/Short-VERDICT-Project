function varargout = show_geom3D(data, geom, dsl, hf)
%
% [hf, hs] = show_geom3D(data, geom, dsl)
%            show_geom3D(data2, geom2, dsl2, hf)
%
%  hf - figure handle
%  hs - surf handle (can be changed manually)
%
%  hs.FaceAlpha = 0.5 ;  % changes transparency of surface
%  delete(hs)
%
%



if ndims(data) > 3
    error(['Data can only be 3D.'])
end

if ~isreal(data)
    warning(['Complex input data - using modulus'])
    data = abs(data) ;
end

data = mat2gray(data) ;

nsl = length(geom) ;
if dsl > nsl
    error(['Slice number greater than number of slices.'])
end

[n1, n2, n3] = size(data) ;

X = zeros([n1 n2]) ;
Y = zeros([n1 n2]) ;
Z = zeros([n1 n2]) ;

% confusing x,y row, column. Better to return to using surf with care over
% which coord is last due to the way they colour the faces

    for iy = 1:n1+1
        for ix = 1:n2+1
            coord = geom(dsl).IPP + ...
                (ix-1.5) * geom(dsl).PixelSpacing_HW(2) * geom(dsl).IOP(1:3) + ...
                (iy-1.5) * geom(dsl).PixelSpacing_HW(1) * geom(dsl).IOP(4:6) ;
            X(iy,ix) = coord(1) ;
            Y(iy,ix) = coord(2) ;
            Z(iy,ix) = coord(3) ;
            
        end
    end

if nargin > 3       
   figure(hf) ;
   hold on
else
   hf = figure('Name','surf plot') ;
end

hs = surf(X,Y,Z,data(:,:,dsl),'EdgeColor','None') ;
colormap('gray'), xlabel('L'), ylabel('P'), zlabel('H')
axis image

if nargout > 0
    varargout{1} = hf ;
    varargout{2} = hs ;
end



