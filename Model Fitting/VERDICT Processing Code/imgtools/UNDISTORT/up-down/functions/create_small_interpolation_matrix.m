%using weight=1-abs(dist, also assign with Jacobian
function [interpolation_matrix1, im2_est]=create_small_interpolation_matrix(reference_image_cycle,Vx,Vy)
im1=reference_image_cycle;
row=size(im1,1);
col=size(im1,2);

       interpolation_matrix1=zeros(row*col,row*col);  
       
      
[row col]=size(Vx);
Vx1=floor(Vx);
Vx2=ceil(Vx);
Vy1=floor(Vy);
Vy2=ceil(Vy);
S = sparse(1,1,0,row*col,row*col);
row_ind=1;
for jj=1:col
for ii=1:row
        weights=[];
        c=[];
        d=[];
        ind=[];
        if ii+Vx1(ii,jj) >=1 && ii+Vx1(ii,jj) <= row && jj+Vy1(ii,jj) >=1 && jj+Vy1(ii,jj) <= col
       % c=[c 1/sqrt( (Vx1(ii,jj)-Vx(ii,jj))^2 + (Vy1(ii,jj)-Vy(ii,jj))^2 )];
       temp1=1-abs(Vx1(ii,jj)-Vx(ii,jj));
       temp2=1-abs(Vy1(ii,jj)-Vy(ii,jj));
       c=[c temp1*temp2];
        d=[d im1(ii+Vx1(ii,jj),jj+Vy1(ii,jj))];
        ind=[ind (jj+Vy1(ii,jj)-1)*row+(ii+Vx1(ii,jj))];
        end
        if ii+Vx1(ii,jj) >=1 && ii+Vx1(ii,jj) <= row && jj+Vy2(ii,jj) >=1 && jj+Vy2(ii,jj) <= col
       % c=[c 1/sqrt( (Vx1(ii,jj)-Vx(ii,jj))^2 + (Vy2(ii,jj)-Vy(ii,jj))^2 )];
        temp1=1-abs(Vx1(ii,jj)-Vx(ii,jj));
        temp2=1-abs(Vy2(ii,jj)-Vy(ii,jj));
        c=[c temp1*temp2];
        d=[d im1(ii+Vx1(ii,jj),jj+Vy2(ii,jj))];
        ind=[ind (jj+Vy2(ii,jj)-1)*row+(ii+Vx1(ii,jj))];
        end
        if ii+Vx2(ii,jj) >=1 && ii+Vx2(ii,jj) <= row && jj+Vy1(ii,jj) >=1 && jj+Vy1(ii,jj) <= col
        %c=[c 1/sqrt( (Vx2(ii,jj)-Vx(ii,jj))^2 + (Vy1(ii,jj)-Vy(ii,jj))^2 )];
         temp1=1-abs(Vx2(ii,jj)-Vx(ii,jj));
         temp2=1-abs(Vy1(ii,jj)-Vy(ii,jj));
         c=[c temp1*temp2];
        d=[d im1(ii+Vx2(ii,jj),jj+Vy1(ii,jj))];
          ind=[ind (jj+Vy1(ii,jj)-1)*row+(ii+Vx2(ii,jj))];
        end
        if ii+Vx2(ii,jj) >=1 && ii+Vx2(ii,jj) <= row && jj+Vy2(ii,jj) >=1 && jj+Vy2(ii,jj) <= col
       % c=[c 1/sqrt( (Vx2(ii,jj)-Vx(ii,jj))^2 + (Vy2(ii,jj)-Vy(ii,jj))^2 )];
        temp1=1-abs(Vx2(ii,jj)-Vx(ii,jj));
       temp2=1-abs(Vy2(ii,jj)-Vy(ii,jj));
       c=[c temp1*temp2];
        d=[d im1(ii+Vx2(ii,jj),jj+Vy2(ii,jj))];
          ind=[ind (jj+Vy2(ii,jj)-1)*row+(ii+Vx2(ii,jj))];
        end
        if length(d)<1 | length(find(c))<1
            im2_est(ii,jj)=im1(ii,jj);
             S(row_ind,(jj-1)*row+(ii))=1;
        else
            c=c./sum(c);
            S(row_ind,ind)=c;
         
            
            im2_est(ii,jj)=sum(c.*d);
        end
        row_ind=row_ind+1;
    end
    
end
interpolation_matrix1=S;
      end
     
