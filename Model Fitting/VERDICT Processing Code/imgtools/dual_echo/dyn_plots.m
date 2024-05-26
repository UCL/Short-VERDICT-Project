function dyn_plots(vdyn,sl,mdyn, vhighres, mhighres) 
% DYN_PLOTS Plots of dynamic changes in subplots.
% Left click to place centre of ROI
% 
% dyn_plots(vdyn,sl)
% dyn_plots(vdyn,sl, mdyn, vhighres, mhighres)
%
% vdyn [ny nx nz ndyn necho]
% sl slice number
%  mdyn structure with geom structure for placing ROI on alternate figure
%  (eg a high res T2 image) as output from d2mat
%  vhighres [nhy nhx nz] usually created using dreslice to match planes
%  of vdyn
%  mhighres structure containing geom structure
%
% Example
%
% ddyn = datparse(dselect) ;
% [vdyn, mdyn] = d2mat(ddyn,{'slice','dyn','echo'},'op','fp') ;
% dt2 = datparse(dselect) ;
% [vt2, mt2] = d2mat(dt2,{'slice'},'op','fp') ;
% [vt2r, mt2r] = dreslice(vt2, mt2, mdyn,'PixelSpacing','input') ;
% dyn_plots(vdyn,3,mdyn, vt2r, mt2r)
%
%
% D.Atkinson@ucl.ac.uk
%
% See also D2MAT
% 

if nargin> 2
    ishighres = true;
    himhigh = figure ;
    himhighim = imshow(vhighres(:,:,sl),[], ...
        'XData',mhighres.geom(sl).XData, 'YData',mhighres.geom(sl).YData) ;
    himhighax = gca ;
else
    ishighres = false ;
end
    
him = figure ;
himshow = imshow(squeeze(vdyn(:,:,sl,end,1)),[]);
himax = gca;
set(himshow,'ButtonDownFcn',@local_plots) ;

    function local_plots(~,~)
        currP = get(himax,'CurrentPoint') ;
        centp(1) = currP(1,1) ;  
        centp(2) = currP(1,2) ;
        
        centp = round(centp) ;
        cent(2) = centp(1) ; cent(1) = centp(2) ;
        
        m=5 ; % rows
        n=5 ; % cols
        
        rowoff = [-ceil((m-1)/2) : m-ceil((m-1)/2)] ;
        coloff = [-ceil((n-1)/2) : n-ceil((n-1)/2)] ;
        obw = 1/n ;
        obh = 1/m ;
        
        % lower-left, lower right, upper right, upper left, lower-left
        xx = [cent(2)+coloff(1)-0.5 cent(2)+coloff(n)+0.5 cent(2)+coloff(n)+0.5 cent(2)+coloff(1)-0.5 cent(2)+coloff(1)-0.5] ;
        yy = [cent(1)+rowoff(m)+0.5 cent(1)+rowoff(m)+0.5 cent(1)+rowoff(1)-0.5 cent(1)+rowoff(1)-0.5 cent(1)+rowoff(m)+0.5] ;
        
        line(xx,yy,'LineWidth',2,'Parent',himax);
        
        % Second Image
        if ishighres
            xxh = convData(xx,[1 size(vdyn,2)],mdyn.geom(sl).XData) ;
            yyh = convData(yy,[1 size(vdyn,1)],mdyn.geom(sl).YData) ;
            
            line(xxh, yyh, 'LineWidth',2,'Parent',himhighax) ;
        end
       
        
        dat = vdyn(cent(1)+rowoff,cent(2)+coloff,sl,:,:);
        mind = min(dat(:));
        maxd = max(dat(:)) ;
        
        hf = figure('Name',['Centre (row, col): (',num2str(cent(1)),...
            ', ',num2str(cent(2)),')']) ;
        
        for p=1:m*n
            row = ceil(p/n) ;
            col = ceil((p-(row-1)*n)) ;
            
            r = cent(1)+rowoff(row);
            c = cent(2)+coloff(col) ;
            % ha = subplot(m,n,p) ;
            ha = axes('Position',[obw*(col-1) 1-obh-(obh*(row-1))  obw obh]) ;
            plot(squeeze(vdyn(r,c,sl,:,1)))
            hold on
            plot(squeeze(vdyn(r,c,sl,:,2)))
            axis([0 size(vdyn,4) mind maxd])
            
            set(ha,'XTickLabel',{})
            set(ha,'YTickLabel',{})
            %set(ha,'OuterPosition',[obw*(col-1) 1-obh-(obh*(row-1))  obw-aeps obh-aeps]) ;
        end
    end
end


