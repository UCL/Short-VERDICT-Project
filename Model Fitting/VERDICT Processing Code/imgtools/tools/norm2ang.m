function norm2ang
% NORM2ANG Plane normals to angulations (Philips)

% Corners of cubes (will be nornmalised later)
p(1).normal = [ 1  1 -1] ;
p(2).normal = [-1  1 -1] ;
p(3).normal = [ 1 -1 -1] ;
p(4).normal = [-1 -1 -1] ;

% Pick in-plane vectors by finding one that is perpendicular to both the
% normal and, say, [0 1 0]

refv = [ 0 1 0 ];

for ip = 1:length(p)
    nnorm = norm(p(ip).normal) ;
    p(ip).normal = p(ip).normal ./ nnorm ;
    
    p(ip).v1u = cross(p(ip).normal, refv) ;
    p(ip).v1u = p(ip).v1u ./ norm(p(ip).v1u) ;
    
    p(ip).v2u = cross(p(ip).normal, p(ip).v1u) ;
    
    iop = [ p(ip).v1u  -p(ip).v2u] ;
    
    [ang_LPH, ORI] = dgeom(iop)
end


