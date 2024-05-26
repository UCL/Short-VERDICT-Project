function tshow_keyp
% tshow_keyp called by tshow figure
% Mappings:
%  a make current point apex (cardiac)
%  b make current point base (cardiac)
%  c camera orbit (see numbers below)
%  e,E ellipsoid (delete with J)
%  f  set manual track point
%  F  draws manual track
%  h,H Help
%  j keep (protect) ellipsoids
%  J delete unprotected ellipsoids
%  k keep (protect) tracks
%  K keep (protect) images
%  l draw long axis
%  m,M montage (with FA if M)
%  P peanut at current point
%  q report current point
%  r ribbon
%  R seed over region
%  s set seed to current point
%  t, T track (T for one way only, otherwise fwd and back)
%  x   delete unprotected tracks
%  X   delete unprotected images
%
%  0 orbit off
%  1,2,3,4,5,6,7,8,9  store camera posn for orbit
%
%
%
% seems like it has to be a separate fcn due to bug (?) in way callbacks
% get handled.
%
% David Atkinson  D.Atkinson@ucl.ac.uk
%
% $Id: tshow_keyp.m 215 2008-11-21 16:17:15Z ucacdat $
%

dfig = gcbf ;
UD = get(dfig,'UserData') ;
handles = guihandles(UD.cfig) ;

currc = get(dfig,'CurrentCharacter') ;
switch currc
    case {'a'} % apex (cardiac)
      currp = get(findobj(dfig,'type','axes'),'CurrentPoint') ;
      cprcs = currp_rcs(currp,UD,handles) ;  
      cpxyz = UD.rcs2xyz * [cprcs(1:3) ; 1] ;
      UD.apex_xyz = cpxyz(1:3) ;
      set(dfig,'UserData',UD)
    case {'b'} % base (cardiac)
      currp = get(findobj(dfig,'type','axes'),'CurrentPoint') ;
      cprcs = currp_rcs(currp,UD,handles) ;  
      cpxyz = UD.rcs2xyz * [cprcs(1:3) ; 1] ;
      UD.base_xyz = cpxyz(1:3) ;
      set(dfig,'UserData',UD)
    case{'l'} % draw long axis
        if isfield(UD,'apex_xyz') && isfield(UD,'base_xyz')
            hla = plot3([UD.apex_xyz(1) UD.base_xyz(1)], ...
                        [UD.apex_xyz(2) UD.base_xyz(2)], ...
                        [UD.apex_xyz(3) UD.base_xyz(3)] ) ;
        else
            disp(['Set axis and base'])
        end
    case {'0' }
        UD.orbitflag = 0 ;
        UD.maxorb = 0 ;
        set(dfig,'UserData',UD)
    case {'1','2','3','4','5','6','7','8','9'}
        UD.orbitflag = 1 ;
        UD.orbitpos(str2num(currc),:) = campos ;
        UD.orbitup(str2num(currc),:) = camup ;
        UD.maxorb = max(UD.maxorb, str2num(currc)) ;
        set(dfig,'UserData',UD)
    case {'h','H' } % help
        help tshow_keyp
    case {'X'} % delete surfaces
        hs = findobj(dfig,'Tag','surf') ;
        delete(hs)
    case {'c' } % camera orbit
        nsub = 50 ;
        iframe = 1 ;
        if UD.orbitflag == 1
            % loop back to beginning
            UD.orbitpos(UD.maxorb + 1,:) = UD.orbitpos(1,:) ; 
            UD.orbitup(UD.maxorb + 1,:) = UD.orbitup(1,:) ; 
          for ipos = 2:(UD.maxorb + 1)
            src = UD.orbitpos(ipos-1,:) ;
            dest = UD.orbitpos(ipos,:) ;
            delt = (dest-src)/nsub ;
            
            usrc = UD.orbitup(ipos-1,:) ;
            udest = UD.orbitup(ipos,:) ;
            udelt = (udest-usrc)/nsub ;
            
            for isubpos = 1:nsub+1
                cp = src + (isubpos-1)*delt ;
                campos(cp)
                cu = usrc + (isubpos-1)*udelt ;
                camup(cu)
                drawnow
                set(gcf,'Units','pixels')
                fpos = get(gcf,'Position') ;
                fpos(1) = 1 ; fpos(2) = 1 ;
                M(iframe) = getframe(gcf,fpos) ;
                iframe = iframe + 1 ;
            end
          end
         
          [mfilename, mpathname] = uiputfile('*.avi', 'Save avi as');
          movie2avi(M,fullfile(mpathname,mfilename),'fps',15, ...
                                               'compression','None')
          
        else
            disp(['orbit flag not set'])
        end
        
    case {'x'}
        % delete tracks
        hs = findobj(dfig,'Tag','track') ;
        delete(hs)
    case {'s'}
        % set seed coordinate to currp
        currp = get(findobj(dfig,'type','axes'),'CurrentPoint') ;
        UD.seed = currp_rcs(currp,UD,handles) ;
        set(dfig,'UserData',UD)
    case {'q'} % report current point
        disp(' ')
        currp = get(findobj(dfig,'type','axes'),'CurrentPoint') ;
        
        cstart_rcs = currp_rcs(currp,UD,handles) ;
        D = interptensor(UD.D,cstart_rcs) ; % tensor at current point 
        [U,S,V] = svd(D);
        v = U(:,1); % primary ev in xyz
        
        seed = currp_rcs(currp,UD,handles) ;
        
        if isfield(UD,'apex_xyz') && isfield(UD,'base_xyz')
            cmap = jet ;
           pix_xyz = UD.rcs2xyz * [seed(:);1] ;
           [ha, inca] = helical(v, pix_xyz(1:3), UD.apex_xyz, UD.base_xyz);
           cind = round(1+(ha+pi/2)/pi*(size(cmap,1)-1)) ;
           col= cmap(cind,:);
           disp(['Helical angle ',num2str(ha),'  inclination ',num2str(inca)])
           disp(['Color index ',num2str(cind),'  color [',num2str(col),']'])
           
        end
        [evec, eval] = d2eig(D) ;
        disp(['Current point rcs ',num2str(seed(1)),' ',num2str(seed(2)), ...
            ' ',num2str(seed(3))])
        disp(['EV1 xyz ',num2str(v(1)),' ',num2str(v(2)),' ',num2str(v(3))])
        disp(['Evals ',num2str(eval(1)),' ',num2str(eval(2)),' ',num2str(eval(3))]) ,...
        disp(['Norm evals ',num2str(eval(1)./eval(3)),' ',num2str(eval(2)/eval(3)),'  1.'])    
    case {'P'} % peanut at current point
        currp = get(findobj(dfig,'type','axes'),'CurrentPoint') ;
        cprcs = currp_rcs(currp,UD,handles) ;
        cprcs = round(cprcs) ;
        coordstr = [num2str(cprcs(1)),',', ...
            num2str(cprcs(2)),',',num2str(cprcs(3))] ;
        disp(['suggest: peanut(B0(',coordstr,'),DWI(',coordstr,',:), grad, bnz)'])
    case {'J'}
        % delete ellipsoids
        hs = findobj(dfig,'Tag','ellipsoid') ;
        delete(hs)    
    case {'K' } % keep current surfaces 
        hs = findobj(dfig,'Tag','surf') ;
        set(hs,'Tag','surf-nodelete') ;
    case {'j' } % keep current ellipsoids 
        hs = findobj(dfig,'Tag','ellipsoid') ;
        set(hs,'Tag','ellipsoid-nodelete') ;
    case {'k' } % keep current tracks 
        hs = findobj(dfig,'Tag','track') ;
        set(hs,'Tag','track-nodelete') ; 
    case {'m', 'M' } % m montage, M montage plus FA colour coded
        % similar code to fig_draw in tshow.m
        dplane = get(handles.popupmenu3,'Value') ;
        vslice = UD.slice ;
        
        ll(1) = str2double(get(handles.edit9,'String')) ;
        ll(2) = str2double(get(handles.edit7,'String')) ;
        ll(3) = str2double(get(handles.edit11,'String')) ;

        ul(1) = str2double(get(handles.edit10,'String')) ;
        ul(2) = str2double(get(handles.edit8,'String')) ;
        ul(3) = str2double(get(handles.edit12,'String')) ;
        
        slcs = [ll(dplane):ul(dplane)] ;
        hw = waitbar(0,'Generating montage data') ;
        
        if strcmp(currc,'M') == 1
            nsl = 2 ;
        else
            nsl = 1 ;
        end
        for islice = 1:length(slcs)
            UD.slice = slcs(islice) ;
            img = tshow_comp_image(dfig, UD, handles)  ;
            if nsl == 2
                vs = get(handles.popupmenu1,'Value') ;
                set(handles.popupmenu1,'Value',2) ;
                img2 = tshow_comp_image(dfig, UD, handles)  ;
                set(handles.popupmenu1,'Value',vs) ;
            end
            for id = 1:3
              indim{id} = [ll(id):ul(id)] ;
            end
            indim{dplane} = 1 ; % returned as one slice 

            if ndims(img) == 4
              indim{4} = [1:3] ;
              img = squeeze(img(indim{:})) ;
              mont(:,:,:,(islice-1)*nsl+1) = permute(img,[1 2 4 3]) ;
              if nsl == 2
                 img2 = squeeze(img2(indim{:})) ;
                 mont(:,:,:,(islice-1)*nsl+2) = permute(img2,[1 2 4 3]) ;
              end
            else
                img = squeeze(img(indim{:})) ;
                mont(:,:,1,islice) = img ;
            end
            waitbar(islice/(length(slcs)),hw) 
        end % slice loop
        close(hw)
        
        UD.slice = vslice ;
        set(dfig,'UserData',UD) ;
        
        contents = get(handles.popupmenu1,'String') ;
        select = contents{get(handles.popupmenu1,'Value')} ;
        figure('Name',['tshow ',select])
        
        if dplane ~= 3
            mont = rotn90(mont,1) ;
        end
        montage(mont)
      
    case { 'e','E' } % ellipsoid
        % copy code from track for getting current point in RCS
        currp = get(findobj(dfig,'type','axes'),'CurrentPoint') ;
        cstart_rcs = currp_rcs(currp,UD,handles) ;
         
        D = interptensor(UD.D,cstart_rcs) ; % tensor at current point 
        
        % adapted from Philip Batchelor's tensorglyph
        [X0,Y0,Z0] = sphere(10);
        D = squeeze(D) ;
        [VC,VL] = eig(D) ;
        [E,idx] = sort(diag(VL)) ;
        VC = VC(:,idx) ;
        E(E<0) = 0 ;
        %E = sqrt(E) ; % differs from PGB
        E = E.*E ;
        sf = (2000).^2;
        
        % fixed size now
        E = E./norm(E) ;  sf = 5 ;
        
        [X,Y,Z] = surftransform(E(1)*X0,E(2)*Y0,E(3)*Z0, VC) ;
        cstart_xyz = UD.rcs2xyz * [cstart_rcs(1) ;
                                   cstart_rcs(2) ;
                                   cstart_rcs(3) ;
        
                                   1];
        
        X = sf*X + cstart_xyz(1) ;
        Y = sf*Y + cstart_xyz(2) ;
        Z = sf*Z + cstart_xyz(3) ;
        surf(X,Y,Z,0.5*ones(size(Z)),'EdgeColor',[0 0 0],'Tag','ellipsoid') ;
    
    case {'t', 'T' } % track (T for one way only)
        currp = get(findobj(dfig,'type','axes'),'CurrentPoint') ;
        cstart_rcs = currp_rcs(currp,UD,handles) ;
        %disp(['UD.slice ',num2str(UD.slice),...
        %   ' f ',num2str(f)])
        %disp(['track start r,c,s ',num2str(cstart_rcs(1)),' ', ...
        %    num2str(cstart_rcs(2)),' ',num2str(cstart_rcs(3)) ])
        
        u0 = [0 0 1]';
        %tra = transpose(ft(UD.D, cstart([2 1 3]), u0)) ;
        tra = transpose(ft(UD.D, cstart_rcs, u0, UD.Rg2rcs)) ;
        
        switch currc
            case 't'
              u0 = [ 0 0 -1]' ;
              %trb = transpose(ft(UD.D, cstart([2 1 3]), u0)) ;
               trb = transpose(ft(UD.D, cstart_rcs, u0, UD.Rg2rcs)) ;
        
               tr = cat(1, trb([end:-1:2],:), tra) ;
            otherwise
                tr = tra ;
        end
        
        
        disp(['  track length ',num2str(size(tr,1))])
        hl = plot_track(tr,UD,handles) ;
        set(hl,'Tag','track')
        
        UD.tr = tr ;
        UD.seed = cstart_rcs ;
        set(dfig,'UserData',UD) 
    case {'f'} % manual track - add point
        if ~isfield(UD,'q')
            UD.q = [] ;
        end
        q = UD.q ;
        cp = currp_rcs(get(findobj(dfig,'type','axes'),'CurrentPoint'),UD,handles ) ;
        q = cat(1,q,cp') ;
        UD.q =  q ;
        hq = findobj(dfig,'Tag','q') ;
        delete(hq)
        hq = plot_track(q,UD,handles);
        set(hq,'Tag','q')
        set(dfig,'UserData',UD)
    case {'F'} % make track from q
        if ~isfield(UD,'q')
            UD.q = [] ;
        end
        hq = findobj(dfig,'Tag','q') ;
        set(hq,'Tag','track')
        tr = UD.q ;
        UD.q = [] ;
        UD.tr = tr ;
        set(dfig,'UserData',UD)
    case {'R'} % seed over area
        for sx = -1.0 : 0.5: 1
            for sy =  -1 : 0.5: 1
                st = UD.seed +[ sy sx 0]' ; % in rcs
                u0 = [0 0 1]';
                tra = transpose(ft(UD.D, st, u0, UD.Rg2rcs)) ;
                u0 = [ 0 0 -1]' ;
                trb = transpose(ft(UD.D, st, u0, UD.Rg2rcs)) ;
        
                tr = cat(1, trb([end:-1:2],:), tra) ;
                
                hl = plot_track(tr,UD,handles);
                set(hl,'Tag','track')
            end
        end
        UD.tr = tr ;
        set(dfig,'UserData',UD)
    case {'z'}  % autotrack
       dplane = get(handles.popupmenu3,'Value') ;
       for id = 1:3
         indim{id} = [1:size(UD.D,id)] ;
       end
       indim{dplane} = UD.slice ;
       D = UD.D(indim{:},:,:) ;
       img = zeros(size(D,1),size(D,2), size(D,3)) ;
       for id1 = 1:size(D,1)
           for id2 = 1:size(D,2)
               for id3 = 1:size(D,3)
                 [evec, evals] = d2eig(D(id1,id2,id3,:,:)) ;
                 if sum(evals) > realmin && isfinite(sum(evals))
                    HFAP =1 - abs(dot(evec(:,1)./norm(evec(:,1)),[1 0 0]')) ;
                    LR = abs(dot(evec(:,3)./norm(evec(:,3)),[1 0 0]')) ;
                    ob  = ((evals(2)/evals(3))-1) ;
                    if ob>2 && HFAP > 0.8 && LR > 0.8 
                       st = [id1 id2 id3] ;
                       st(dplane) = UD.slice ;
                       B0 = UD.B0 ;
                       %if B0(st(1),st(2),st(3)) > max(B0(:))/15 
                         u0 = [0 0 1]';
                         tra = transpose(ft(UD.D, st, u0, UD.Rg2rcs)) ;
                         u0 = [ 0 0 -1]' ;
                         trb = transpose(ft(UD.D, st, u0, UD.Rg2rcs)) ;
                         tr = cat(1, trb([end:-1:2],:), tra) ;    
                         hl = plot_track(tr,UD,handles);
                         set(hl,'Tag','track') 
                      % end
                    end
                 end
               end
           end
       end
    case {'r' }  % ribbon
        rwid = 10 ; % ribbon width
        if isfield(UD,'tr')
            tr = UD.tr ;
            
            SRCS = [tr(:,1)' ;
                        tr(:,2)' ;
                        tr(:,3)' ;
                        ones([1 size(tr,1)]) ] ;
            
            SXYZ = UD.rcs2xyz * SRCS ;
            SX = SXYZ(1,:) ;
            SY = SXYZ(2,:) ;
            SZ = SXYZ(3,:) ;
        
            
            for iv = 1:size(tr,1)-1
                trp = tr(iv,:) ;
                if floor(trp(3)) > 0 % pesky edge effects
                  Dv = interptensor(UD.D,tr(iv,:)) ;
                  [ev, lambda] = eig(squeeze(Dv)) ;
                  [evals, ord] = sort([lambda(1,1), lambda(2,2), lambda(3,3)],...
                    2,'descend') ;
                  evecs = [ev(:,ord(1)), ev(:,ord(2)), ev(:,ord(3))];
               
                  [ob, e2v, col] = oblate_info(evecs, evals) ;
               
                  Dv = interptensor(UD.D,tr(iv+1,:)) ;
                  [ev, lambda] = eig(squeeze(Dv)) ;
                  [evals, ord] = sort([lambda(1,1), lambda(2,2), lambda(3,3)],...
                      2,'descend') ;
                  evecs = [ev(:,ord(1)), ev(:,ord(2)), ev(:,ord(3))];
               
                  [ob1, e2v1, col1] = oblate_info(evecs, evals) ;
               
                  % tr is in rcs, eigenvectors are in xyz.
                  % patch will plot in xyz so convert tr to xyz
                  %
                  % pv(1,:) = tr(iv,:) + ob*rwid*e2v' ;
                  % pv(2,:) = tr(iv,:) - ob*rwid*e2v' ;
                  % pv(3,:) = tr(iv+1,:) - ob1*rwid*e2v1' ;
                  % pv(4,:) = tr(iv+1,:) + ob1*rwid*e2v1' ;
                  
                  pv(1,:) = [SX(iv) SY(iv) SZ(iv)] + ob*rwid*e2v' ;
                  pv(2,:) = [SX(iv) SY(iv) SZ(iv)] - ob*rwid*e2v' ;
                  pv(3,:) = [SX(iv+1) SY(iv+1) SZ(iv+1)] - ob1*rwid*e2v1' ;
                  pv(4,:) = [SX(iv+1) SY(iv+1) SZ(iv+1)] + ob1*rwid*e2v1' ;
                 
                  
                  cv = [ col ; col ; col1; col1] ;
                  cv = reshape(cv,[4 1 3]) ;
               
                  patch(pv(:,1), pv(:,2), pv(:,3),cv,...
                      'EdgeColor','none','Tag','track')
                end
            end
        end
        
end

% -----------------------------------------------------------
function hl = plot_track(tr,UD,handles)
col = [str2double(get(handles.edit4,'String')) ...
       str2double(get(handles.edit5,'String')) ...
       str2double(get(handles.edit6,'String'))] ;
        
% line plots in xyz, tr is in rcs
SRCS = [tr(:,1)' ;
        tr(:,2)' ;
        tr(:,3)' ;
        ones([1 size(tr,1)]) ] ;
            
 SXYZ = UD.rcs2xyz * SRCS ;
 SX = SXYZ(1,:) ;
 SY = SXYZ(2,:) ;
 SZ = SXYZ(3,:) ;
        
 hl = line(SX,SY,SZ,'Color',col,'LineWidth',1.5);
 return
 
 % -----------------------------------------------------------
 function cp_rcs = currp_rcs(currp,UD,handles)
 SRCS = UD.xyz2rcs * [currp(1,1) currp(2,1) ; % front_x back_x
                      currp(1,2) currp(2,2) ;
                      currp(1,3) currp(2,3) ;
                   1           1          ] ;
                         
  currp_front_rcs = SRCS(1:3,1) ;
  currp_back_rcs = SRCS(1:3,2) ;
        
  dplane = get(handles.popupmenu3,'Value') ;
       
  f = (UD.slice - currp_front_rcs(dplane))/...
              (currp_back_rcs(dplane)-currp_front_rcs(dplane)) ;
         
  cp_rcs = currp_front_rcs(:) + f*(currp_back_rcs(:)-currp_front_rcs(:)) ;
        