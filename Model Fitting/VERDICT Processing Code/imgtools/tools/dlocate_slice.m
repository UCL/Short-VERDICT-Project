function dlocate_slice(geom)

nsl = length(geom) ;

disp(['Pixel Spacing: ',num2str(geom(1).PixelSpacing_HW(:)')])
disp(['Half Pixel Spacing: ',num2str(geom(1).PixelSpacing_HW(:)'/2.0)])

for isl = 1: nsl
    iop = geom(isl).IOP ;
    ipp = geom(isl).IPP ;
    plane = set_plane('pvv',ipp,iop(1:3),iop(4:6)) ;
    d = point_plane([0 0 0],plane) ; %Patient coord origin to plane distance
    oip = d*plane.normal ; %closest point in plane of origin 
    
    DC_lph = ipp + (DCgp([1:geom(isl).Width],2)-1)*geom(isl).PixelSpacing_HW(2)*iop(1:3) + ...
        (DCgp([1:geom(isl).Height],2)-1)*geom(isl).PixelSpacing_HW(1)*iop(4:6) ;
     
    disp(['Sl: ', num2str(isl),' sl offset: ',num2str(d), ...
        ' DCoffset (row): ',num2str(dot((oip-DC_lph),iop(1:3))), ...
        ' DCoffset (col): ',num2str(dot((oip-DC_lph),iop(4:6)))   ])
end