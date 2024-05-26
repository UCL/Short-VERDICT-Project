function img = tshow_comp_image(fig, UD, handles)
% tshow_comp_image
%
% img = tshow_comp_image(fig, UD, handles)
%
% called from tshow or tshow_keyp
%
% David Atkinson D.Atkinson@ucl.ac.uk
% $Id: tshow_comp_image.m 275 2010-02-06 22:33:05Z ucacdat $
%

contents = get(handles.popupmenu1,'String') ;
select = contents{get(handles.popupmenu1,'Value')} ;

dplane = get(handles.popupmenu3,'Value') ;

for id = 1:3
  indim{id} = [1:size(UD.D,id)] ;
end
indim{dplane} = UD.slice ;

rbv = get(handles.radiobutton1,'Value') ;
if  rbv == 1
    mask = UD.B0(indim{:}) ;
   
    B0max = str2double(get(handles.edit3,'String')) ;
    %loc = find(mask< max(max(max(mask)))/15) ;
    loc = find(mask < B0max) ;
    mask = ones([size(mask,1) size(mask,2) size(mask,3)]) ;
    mask(loc) = 0 ;
else
    mask = ones([ length(indim{1}) length(indim{2}) length(indim{3})]) ;
end
switch select
    case 'FA grayscale'
       v = invariantsb(UD.D(indim{:},:,:),'fa') ;
       img = mat2gray(v,[0 1]) ;
       
    case 'FA colour coded l1'
       D = UD.D(indim{:},:,:) ;
       img = d2cfa(D) ;
       
     case 'FA colour coded l3'
       D = UD.D(indim{:},:,:) ;
       fa = invariantsb(D,'fa') ;
       vec = zeros(size(D,1), size(D,2),size(D,3), 3) ;
       for id1 = 1:size(D,1)
           for id2 = 1:size(D,2)
               for id3 = 1:size(D,3)
                 [ev,lambda] = eig(squeeze(D(id1,id2,id3,:,:)));
                 [evals, ord] = sort([lambda(1,1), lambda(2,2), lambda(3,3)],...
                     2,'descend') ;
               
                 vec(id1,id2,id3,:) = reshape(ev(:,ord(3)),[1 1 1 3]);
               end
           end
       end
       img = vec2col(vec(:,:,:,[1 2 3]),fa,0,1) ; 
    case 'l1.HF * l3.LR'
       D = UD.D(indim{:},:,:) ;
       img = zeros(size(D,1),size(D,2), size(D,3)) ;
       for id1 = 1:size(D,1)
           for id2 = 1:size(D,2)
               for id3 = 1:size(D,3)
                 [evec, evals] = d2eig(D(id1,id2,id3,:,:)) ;
                 if sum(evals) > realmin && isfinite(sum(evals))
                     
                     
                      HF = abs(dot(evec(:,1)./norm(evec(:,1)),[0 0 1]')) ;
                      HFAP =1 - abs(dot(evec(:,1)./norm(evec(:,1)),[1 0 0]')) ;
                      LR = abs(dot(evec(:,3)./norm(evec(:,3)),[1 0 0]')) ;
                      if HF < 0.8
                          HF = 0 ;
                      else
                          HF = 1 ;
                      end
                      if LR<0.8
                          LR = 0 ;
                      else
                          LR = 1 ;
                      end
                      if HFAP < 0.7
                          HFAP = 0 ;
                      else
                          HFAP = 1 ;
                      end
                      ob = ((evals(2)/evals(3))-1) ;
                      
                      img(id1,id2,id3) = ob *HFAP*LR ;
                      %(evals(2)-evals(3))/sum(evals) * ...
                      %      HF*LR;
                 else
                   img(id1,id2,id3) = 0 ;
                 end
               end
           end
       end
       % img = mat2gray(img,[0.1 0.3]) ;
       %eshow(img)
       img = mat2gray(img,[0 1]) ;
       
    case '1/(l1 - l2 + 1)'
       D = UD.D(indim{:},:,:) ;
       img = zeros(size(D,1),size(D,2), size(D,3)) ;
       for id1 = 1:size(D,1)
           for id2 = 1:size(D,2)
               for id3 = 1:size(D,3)
                 [evec, eval] = d2eig(D(id1,id2,id3,:,:)) ;
                 if sum(eval) > realmin && isfinite(sum(eval))
                   img(id1,id2,id3) = 1/(eval(1)-eval(2)+1) /sum(eval);
                 else
                     img(id1,id2,id3) = 0 ;
                 end
               end 
           end
       end
       img = mat2gray(img,[0 600]) ;
       
    case '2(l2 - l3)'
       D = UD.D(indim{:},:,:) ;
       img = zeros(size(D,1),size(D,2),size(D,3)) ;
       for id1 = 1:size(D,1)
           for id2 = 1:size(D,2)
               for id3 = 1:size(D,3)
                 [evec, eval] = d2eig(D(id1,id2,id3,:,:)) ;
                 if sum(eval) > realmin && isfinite(sum(eval))
                   img(id1,id2,id3) = 2*(eval(2)-eval(3)) /sum(eval);
                 else
                   img(id1,id2,id3) = 0 ;
                 end
               end     
           end
       end
       img = mat2gray(img,[0 0.5]) ;
       
    case 'oblate colour coded l3'
       D = UD.D(indim{:},:,:) ;
       ob = zeros(size(D,1),size(D,2),size(D,3)) ;
       vec = zeros(size(D,1),size(D,2),size(D,3),3) ;
       
       for id1 = 1:size(D,1)
           for id2 = 1:size(D,2)
               for id3 = 1: size(D,3)
                 [evec, eval] = d2eig(D(id1,id2,id3,:,:)) ;
                 if sum(eval) > realmin && isfinite(sum(eval))
                   ob(id1,id2,id3) = 2*(eval(2)-eval(3)) /sum(eval);
                 else
                   ob(id1,id2,id3) = 0 ;
                 end
                 vec(id1,id2,id3,:) = reshape(evec(:,3),[1 1 1 3]) ;
               end
                   
           end
       end
       
       img = vec2col(vec(:,:,:,[1 2 3]),ob,0,0.7) ;
    case '(v2/v3 - 1) colour coded l3'
       D = UD.D(indim{:},:,:) ;
       ob = zeros(size(D,1),size(D,2),size(D,3)) ;
       vec = zeros(size(D,1),size(D,2),size(D,3),3) ;
       
       for id1 = 1:size(D,1)
           for id2 = 1:size(D,2)
               for id3 = 1: size(D,3)
                 [evec, eval] = d2eig(D(id1,id2,id3,:,:)) ;
                 if sum(eval) > realmin && isfinite(sum(eval))
                   ob(id1,id2,id3) = (eval(2)/eval(3)) -1;
                 else
                   ob(id1,id2,id3) = 0 ;
                 end
                 vec(id1,id2,id3,:) = reshape(evec(:,3),[1 1 1 3]) ;
               end
                   
           end
       end
       ob(find(ob>2))=1;
       img = vec2col(vec(:,:,:,[1 2 3]),ob,0,2) ;
       %eshow(squeeze(img),'isrgb',1)
        
    case 'Westin rgb'
       D = UD.D(indim{:},:,:) ;
       vec = zeros(size(D,1),size(D,2),size(D,3),3) ;
       for id1 = 1:size(D,1)
           for id2 = 1:size(D,2)
               for id3 = 1: size(D,3)
                 [evec, eval] = d2eig(D(id1,id2,id3,:,:)) ;
                 if sum(eval) > realmin && isfinite(sum(eval))
                   vec(id1,id2,id3,:) = reshape([ (eval(1) - eval(2) )/sum(eval)  ; ...
                       2*(eval(2)-eval(3))/sum(eval) ; ...
                       3*eval(3)/sum(eval)  ],[1 1 3]) ;
                 else
                 vec(id1,id2,id3,:) = 0 ;
                 end
               end
                   
           end
       end
       lo = find(vec<0) ;
       vec(lo) = 0 ;
       vec = mat2gray(vec,[0 1]) ;
       img = vec2col(vec,ones([size(vec,1) size(vec,2) size(vec,3)]),0,1) ;
       
    case 'l3 (-ve red)'
       D = UD.D(indim{:},:,:) ;
       vec = zeros(size(D,1),size(D,2),size(D,3),3) ;
       for id1 = 1:size(D,1)
           for id2 = 1:size(D,2)
               for id3 = 1: size(D,3)
                 [evec, eval] = d2eig(D(id1,id2,id3,:,:)) ;
                 if sum(eval) > realmin && isfinite(sum(eval))
                     if eval(3) > 0
                       vec(id1,id2,id3,:) = reshape([eval(3) eval(3) eval(3)],[1 1 1 3]);
                     else
                       vec(id1,id2,id3,:) = reshape([1 0 0] ,[1 1 1 3]); 
                     end
                 else
                   vec(id1,id2,id3,:) = 0 ;
                 end
               end
                   
           end
       end
       
       vec = mat2gray(vec,[0 1]) ;
       img = vec2col(vec,ones([size(vec,1) size(vec,2) size(vec,3)]),0,1) ;
       
    case 'B0'   
        if isfield(UD,'B0')
            contents3 = get(handles.popupmenu3,'String') ;
            select3 = contents3{get(handles.popupmenu3,'Value')} ;
            img = UD.B0(indim{:}) ;
            B0max = str2double(get(handles.edit3,'String')) ;
          img = mat2gray(img,[0 B0max]) ;
          
        end
    case{'helical angle','inclination angle' }
        if isfield(UD,'apex_xyz') && isfield(UD,'base_xyz')
            cmap = jet(64) ;
            D = UD.D(indim{:},:,:) ;
            fa = invariantsb(D,'fa') ;
            img = zeros(size(D,1), size(D,2),size(D,3), 3) ;
            
            for id1 = 1:size(D,1)
              for id2 = 1:size(D,2)
                for id3 = 1:size(D,3)
                    pix_xyz = UD.rcs2xyz * [indim{1}(id1);indim{2}(id2);indim{3}(id3);1] ;
                  [U,S,V] = svd(squeeze(D(id1,id2,id3,:,:))) ;
                  v = U(:,1) ;
                  [ha, inca] = helical(v, pix_xyz(1:3), UD.apex_xyz, ...
                     UD.base_xyz) ;
                  if strcmp(select,'helical angle')
                    cind = round(1+(ha+pi/2)/pi*(size(cmap,1)-1)) ;             
                    col = cmap(cind,:) ;
                  else % inclination angle
                      cind = round(1+(inca+pi/2)/pi*(size(cmap,1)-1)) ;
                      col = cmap(cind,:) ;
                  end
                  img(id1,id2,id3,:) = reshape(col(:),[1 1 1 3]) ;
                  
                end
              end
            end
           
        else
            warning(['First set long axes using a and b'])
        end
    otherwise
        disp('not implemented')
end
if ndims(img) == 4 
  mask = repmat(mask,[1 1 1 3]) ;
end
img = img .* mask ;

UD.image = img ;
set(fig,'UserData',UD) ;
