function draw_image(fig)
% DRAW_IMAGE  Draws image grid for explaining coord systems and DICOM
%  draw_image(fig_type)
%
% fig_type can be 'intrinsic', 'simplified', or, 'full'
%
% Copyright, 2020, David Atkinson
% D.Atkinson@ucl.ac.uk
%

switch fig
    case 'intrinsic'
        nr = 5 ;
        nc = 5 ;
        theta = 0 ;
        plotLPHax = false ;
        plotIOP = false ;
        plotIPP = false ;
        plotperp = false ;
        tlc = [1 1] ;
        ipp = [ 1 1] ; % put origin back to 1 1
        tspc = 0.2 ;
    case 'simplified'
        nr = 3 ;
        nc = 3 ;
        theta= 20 ;
        plotLPHax = true ;
        plotIOP = true ;
        plotIPP= false ;
        plotperp = true ;
        tlc = [1 1] ;
        ipp = [ 0 0] ;
        tspc = 0.1 ;
    case 'full'
        nr = 4 ;
        nc = 4 ;
        theta= 20 ;
        plotLPHax = true ;
        plotIOP = true ;
        plotIPP = true ;
        plotperp = false ;
        tlc = [1 1] ;
        ipp = [ 2 2] ;
        tspc = 0.2 ;
    otherwise
        error('Unknown figure')
end



col_LPH = [165 42 42]./256 ; % brown

txtb = {'FontSize',14, 'BackgroundColor', [0.9 0.9 0.9] } ;


AR = [cosd(theta) sind(theta) 0;...
     -sind(theta) cosd(theta) 0; ...
      0           0           1] ;
 
AT = [ 1        0        0 ; ...
       0        1        0 ;...
      -tlc(1) -tlc(2)  1 ] ;
  
AIPP = [ 1        0        0 ; ...
         0        1        0 ;...
       ipp(1)   ipp(2)     1 ] ;

tform =  affine2d(AT*AR*AIPP) ;
 
 
% Intrinsic coordinates
figure
hold on, axis equal, grid on
set(gca,'YDir','reverse')
set(gca,'XAxisLocation','top')
title(fig)

% coloumn pixel edges
for icedge = 1:nc+1
    X(:,icedge) = [icedge-0.5 icedge-0.5] ;
    Y(:,icedge) = [0.5 nr+0.5];
end
[XR, YR] = transformPointsForward(tform,X,Y) ;
plot(XR,YR, 'LineWidth', 2, 'Color', [0 0 0])

% row pixel edges
for iredge = 1:nr+1
    X(:,iredge) = [0.5 nc+0.5]; 
    Y(:,iredge) = [iredge-0.5 iredge-0.5] ;
end
[XR, YR] = transformPointsForward(tform,X,Y) ;
plot(XR,YR, 'LineWidth', 2, 'Color', [0 0 0])

iop = [1 1 ; 2 1 ; 1 1 ; 1 2] ; % packed coords of start and end of row arrow then column arrow
[iopr] = transformPointsForward(tform,iop) ;

ps = [nc-0.5 0.3 ; nc+0.5 0.3 ; nc+0.5+0.3 0.5 ; nc+0.5+0.3 1.5] ;
psr = transformPointsForward(tform,ps)  ;


% Axis
if plotLPHax
    p1 = [0 0];                         % First Point
    p2 = [nc 0];                         % Second Point
    dp = p2-p1;                         % Difference
    
    quiver(p1(1),p1(2),dp(1),dp(2),0,'Color',col_LPH, 'LineWidth',2)
    text(p2(1)+tspc, p2(2), 'Left', 'FontSize',14,'Color',col_LPH )
    
    p2 = [0 nr];                         % Second Point
    dp = p2-p1;                         % Difference
    
    quiver(p1(1),p1(2),dp(1),dp(2),0,'Color',col_LPH, 'LineWidth',2)
    text(p2(1)+tspc, p2(2), 'Posterior', 'FontSize',14,'Color',col_LPH )
end

if plotIOP
    % IOP(1:3)
    quiver(iopr(1,1), iopr(1,2), iopr(2,1)-iopr(1,1),iopr(2,2)-iopr(1,2) ,0,'Color', col_LPH, 'LineWidth',2, 'MaxHeadSize',0.5)
    text(iopr(2,1)+tspc, iopr(2,2), 'iop(1:3)',txtb{:} )
    if plotperp
        plot([ iopr(2,1) iopr(2,1)], [iopr(2,2) 0],':', 'LineWidth',2)
    end
    
    % IOP(4:63)
    quiver(iopr(3,1), iopr(3,2), iopr(4,1)-iopr(3,1),iopr(4,2)-iopr(3,2) ,0,'Color', col_LPH, 'LineWidth',2, 'MaxHeadSize',0.5)
    text(iopr(4,1)-tspc, iopr(4,2)+2*tspc, 'iop(4:6)', txtb{:})
    if plotperp
        plot([ iopr(2,1) 0], [iopr(2,2) iopr(2,2)],':', 'LineWidth',2)
    end
end

if plotIPP
    quiver(0,0,ipp(1), ipp(2), 0, 'Color', col_LPH, 'LineWidth',2)
    text(1,1,'ImagePositionPatient', txtb{:})
end


switch fig
    case 'intrinsic'
        plot(1,1,'ro','MarkerFaceColor','r')
        text(1+tspc, 1-tspc, '(1,1)',txtb{:})
        axis([-1 nc+1 -1 nr+1])
        
        xlabel('x'), ylabel('\leftarrow y') 
        set(gca,'FontWeight','bold')
    case 'full'
        plot([psr(1,1) psr(2,1)],[psr(1,2) psr(2,2)],'Color','k','LineWidth',2)
        plot([psr(3,1) psr(4,1)], [psr(3,2) psr(4,2)],'Color','k','LineWidth',2)
        text(psr(1,1)+0.9,psr(1,2),'PixelSpacing(2)',txtb{:})
        text(psr(4,1)+0.2,psr(4,2),'PixelSpacing(1)', txtb{:})
        
end

% iop = [cosd(theta) sind(theta)  0 -sind(theta) cosd(theta) 0] ;
% 
% p1i = [1 0];
% p1lph = p1i * [iop(1:2) ; iop(4:5)]
% % [ L P H ] = [ i j k ] * [ rL rP rH ;
% %                           cL cP cH ;
% %                           sL sP sH ] ;
% 
% 
% p2i = [0 1] ;
% p2lph = p2i * [iop(1:2) ; iop(4:5)]
% dot([1 0 0],iop(4:6))
% dot([0 1 0],iop(4:6))


 



