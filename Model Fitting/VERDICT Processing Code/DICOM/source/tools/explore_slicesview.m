function explore_slicesview(v,m)
% % looking at obiqueslice
% 
% dinfo = getdinfo(dselector) ;
% 
% [v,m] = d2mat(dinfo,{'slice'},'op','fp') ;


hf = figure('WindowScrollWheelFcn',@figScroll, ...
    'Tag','sviewer_fig',...
   'WindowButtonDownFcn',[]) ;  %'WindowButtonDownFcn',@figDown) ;

cmHandle = uicontextmenu(hf) ;
cmHandle.ContextMenuOpeningFcn = @cmopening ;
uimenu(cmHandle,'Label','imcontrast','Callback',@window) ;


ha = axes(hf) ;


% ha.NextPlot = "replacechildren" ;

% set(gcf,'doublebuffer','off');
% imshow(v(:,:,1),[0 2000],'InitialMagnification',200)
% set(gca, 'xlimmode','manual',...
%            'ylimmode','manual',...
%            'zlimmode','manual',...
%            'climmode','manual',...
%            'alimmode','manual');
nslice = size(v,3) ;
dslice = ceil(nslice/2) ;
wlow = 0 ;
whigh = max(v(:)) ;
wstep = (whigh-wlow)/100 ;
cpd = [] ;
inmag = 200 ;
% him = [] ;

an = annotation(hf,'textbox',[0.1 0.1 0.2 0.1],...
            'String',[num2str(wlow), ' ', num2str(whigh),' ',num2str(dslice),'/',num2str(nslice)], ...
            'Color',[1 1 1]) ;

UD.an = an ;
hf.UserData = UD ;

XData = m.geom(1).XData;
YData = m.geom(1).YData;
%warm-up axes
himshow = imshow(v(:,:,1),[],'XData',XData,'YData',YData,'Parent',ha) ;

%ha.Toolbar = [] ;
set(himshow,'Visible','off')

%apply all slices to axes, but make in visible
ip = {'XData',XData,'YData',YData,'CDataMapping','scaled','Visible','off','CData'} ;

for islice = 1:size(v,3)
    him(islice) = image(ha,ip{:},v(:,:,islice)) ;
end

disp(' ')


    function figDown(src,event)
        src.WindowButtonMotionFcn = @figMotion ;
        src.WindowButtonUpFcn = @figUp ;
        cpd = hf.CurrentPoint ;
        UD = src.UserData ;
        UD.an.String = [num2str(wlow), ' ', num2str(whigh),' ',num2str(dslice),'/',num2str(nslice)] ;
        
    end

    function figUp(src,event)
        src.WindowButtonMotionFcn = '' ;
        src.WindowButtonUpFcn = '' ;
        %him.ContextMenu = contextM ;
        
    end

    function figMotion(src,event)
        cp = hf.CurrentPoint ;
        xm = cp(1,1) - cpd(1,1);
        ym = cp(1,2) - cpd(1,2);

        mthresh = 2 ;

        if ym > mthresh
            if whigh > (wlow+wstep)
                whigh = whigh - wstep ;
            end
        elseif ym < -mthresh
            whigh = whigh + wstep ;
        end

        if xm > mthresh
            if wlow >= wstep 
                wlow = wlow - wstep ;
            end
        elseif xm < -mthresh
            if wlow < (whigh-wstep)
                wlow = wlow + wstep ;
            end
        end

%         him = imshow(v(:,:,dslice),[wlow whigh], ...
%             'InitialMagnification',inmag, ...
%             'Parent', ha) ;

%         soff{1} = 'off';
%         cstr = repmat(soff,[1 length(him)]) ;
%         [him.Visible] = deal(cstr{:})  ;
%         him(dslice).Visible = 'on' ;


        hax = him(dslice).Parent ;
        hax.CLim = [wlow whigh] ;
        UD = src.UserData ;
        UD.an.String = [num2str(wlow), ' ', num2str(whigh),' ',num2str(dslice),'/',num2str(nslice)] ;


        cpd = cp ;

    end
 
    function figScroll(src, event)
        if event.VerticalScrollCount > 0
            if dslice < nslice
                dslice = dslice + 1;
            end
        elseif event.VerticalScrollCount < 0
            if dslice > 1
                dslice = dslice - 1;
            end
        end

%         him = imshow(v(:,:,dslice),[wlow whigh], ...
%             'InitialMagnification',inmag, ...
%             'Parent', ha);
%         him.ContextMenu = contextM ;
        soff{1} = 'off';
        cstr = repmat(soff,[1 length(him)]) ;
        [him.Visible] = deal(cstr{:})  ;
        him(dslice).Visible = 'on' ;
        %him(dslice).Parent.CLim = [wlow whigh] ;
        him(dslice).ContextMenu = cmHandle ;
    end

    function window(src, event)
        htool = imcontrast(ha);
        hadjb = findall(htool,'Tag','adjust data button') ;
        hadjb.Visible = 'off' ;
    end
end




function cmopening(src,~)
clickedfig = src.Parent ;
htl = findobj(groot,'Tag','sviewer_fig') ;
hother = setdiff(htl, clickedfig) ;
for iother = 1:length(hother)
    uimenu(src,"Text",['Fig ',hother(iother).Name,' ',num2str(hother(iother).Number)])
end

% For axs linking, use linkaxes. This stores linkprop somewhere and deletes
% local when renewing


% if isempty(htl)
%     src.Children(1).Enable = 'off' ;
%     src.Children(2).Enable = 'off' ;
%     src.Children(3).Enable = 'off' ;
%     src.Children(4).Enable = 'off' ;
%     src.Children(5).Enable = 'off' ;
%     src.Children(6).Enable = 'off' ;
%     src.Children(7).Enable = 'off' ;
% end
end




% for islice = 1:nslice
%     imshow(v(:,:,islice),[0 2000],'InitialMagnification',200)
% end

% s=sliceViewer(v)
% 
% 
% midslice = ceil(nslice/2) ;
% 
% figure
% hold on
% for islice = 1:nslice
%     [L,P,H] = dicom2intrinsic(m.geom(islice),'output', 'LPHcoords') ; 
% 
%     h(islice) = surf(L,P,H,v(:,:,islice),'EdgeColor','None', ...
%     'HandleVisibility','off') ;
% end
% colormap gray
% xlabel('L'),ylabel('P'),zlabel('H')

% For true axial view:
% ha = gca 
% view(ha,[0 0 1])
% ha.CameraUpVector = [0 -1 0]; 
