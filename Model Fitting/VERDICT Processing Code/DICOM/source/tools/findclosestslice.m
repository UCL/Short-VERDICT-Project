function ind_slice = findclosestslice(point3D, geom)
% FINDCLOSESTSLICE Find closest slice to 3D point
% Finds closest slice index to a point (image centre) in 3D LPH.
% Aim is to allow zlinked scrolling even when zoomed and for slices with
% some angulation to each other.
%
% ind_slice = findclosestslice(point3D, geom)
%
% David Atkinson
%
% See also sviewer

arguments
    point3D (1,3) double
    geom
end

nslice = length(geom) ;
d = zeros([1 nslice]) ;

for islice = 1 : nslice
    ipp = geom(islice).IPP ;
    iop = geom(islice).IOP ;

    plane = set_plane('pvv',ipp,iop(1:3),iop(4:6)) ;
    d(islice) = point_plane(point3D,plane) ; %point to plane distance
end

[dsort, ind] = sort(abs(d),'ascend') ;

% below is to avoid jumping due to rounding errors when one volume has
% slices at the location of every otheslice in the other volume
if abs(dsort(1)-dsort(2)) < 0.001 
    ind_slice = min(ind(1), ind(2))  ;
else
    ind_slice = ind(1) ;
end


