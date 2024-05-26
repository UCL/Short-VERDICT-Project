function geom_out = update_spatial_ref(geom_in)
%
% Taken from dgeomextract

geom_out = geom_in ; % improve by checking parameters consistent etc

for isl = 1:length(geom_in)
    iop = geom_in(isl).IOP ;
    ipp = geom_in(isl).IPP ;
    Width = geom_in(isl).Width ;
    Height = geom_in(isl).Height ;
    PS_HW = geom_in(isl).PixelSpacing_HW ;


    plane = set_plane('pvv',ipp,iop(1:3),iop(4:6)) ;
    d = point_plane([0 0 0],plane) ; %Patient coord origin to plane distance
    oip = d*plane.normal ; %closest point in plane of origin 
    oip2ipp = ipp - oip ;
    XData(1) = dot(oip2ipp,iop(1:3)) ;
    YData(1) = dot(oip2ipp,iop(4:6)) ;
    XData(2) = XData(1) + PS_HW(2)*(Width-1) ;
    YData(2) = YData(1) + PS_HW(1)*(Height-1) ;
    
    geom_out(isl).XData = XData ;
    geom_out(isl).YData = YData ;
    
    if exist('imref2d','file') % spatial referencing not in old MATLAB
        [R2D,D] = geom2sro(geom_out(isl)) ;
        geom_out(isl).R2D = R2D;
        geom_out(isl).D = D ;
    end
end

