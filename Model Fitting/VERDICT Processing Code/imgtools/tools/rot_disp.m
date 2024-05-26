

angZ0 = 45 /360*2*pi ;

angY1 = angZ0 ;

RZ0 = makehgtform('zrotate',angZ0) ;
RY1 = makehgtform('yrotate',angY1) ;
RX2 = makehgtform('xrotate',0) ;

veca = [1 0 0 1] ;
vecb = [0 1 0 1] ;
vecc = [0 0 1 1] ;


veca3 = RZ0 * RY1 * RX2* veca' 
vecb3 = RZ0 * RY1 * RX2* vecb'
vecc3 = RZ0 * RY1 * RX2* vecc'

%dot(veca3(1:3),vecc3(1:3))


% PS_HW = [128 128]; Width=128; Height=128;
% offcentres = [0 0 0] ;
% 
% % 1
% ang_LPH = [45 35 0]; ori='TRA'
% [iop, ipp] = dgeom(offcentres, ang_LPH, ori, PS_HW , Width, Height);
% n1 = cross(iop(1:3),iop(4:6))
% 
% % 2
% ang_LPH = [-45 35 0]; ori='TRA'
% [iop, ipp] = dgeom(offcentres, ang_LPH, ori, PS_HW , Width, Height);
% n2 = cross(iop(1:3),iop(4:6))
% 
% % 3
% ang_LPH = [45 0 35]; ori='COR'
% [iop, ipp] = dgeom(offcentres, ang_LPH, ori, PS_HW , Width, Height);
% n3 = cross(iop(1:3),iop(4:6))
% 
% dot(n1,n2)
% dot(n1,n3)
% dot(n2,n3)
% 
% figure
% plot3([0 n1(1)],[0 n1(2)], [0 n1(3)])
% hold on
% plot3([0 n2(1)],[0 n2(2)], [0 n2(3)])
% plot3([0 n3(1)],[0 n3(2)], [0 n3(3)])
% axis square
% axis equal
% 
