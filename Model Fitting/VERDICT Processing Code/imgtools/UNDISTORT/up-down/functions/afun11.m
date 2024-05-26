function [output] = afun11(beta,arg)
% Inverse encoding function with dB0 phase term and coil sensitivities
% 

data_type = @double;

%inputs
S        = data_type(arg.S);
t_s1      = data_type(arg.t_s1);
t_s2     = data_type(arg.t_s2);
dB0_rads = data_type(arg.dB0_rads);
kx       = data_type(arg.k.x);
ky       = data_type(arg.k.y);
u1        = data_type(arg.Sk1);
u2        = data_type(arg.Sk2);
bw_per_pixel=data_type(arg.bw_per_pixel);

[sy,sx,nch,~] = size(u1);

%k-space co-ord system
x_coord = (1:sx) - (sx / 2) - 1; %orignal
y_coord = (1:sy) - (sy / 2) - 1; %original 
% x_coord = (1:sx) - floor(sx / 2) - 1;
% y_coord = (1:sy) - floor(sy / 2) - 1;
x_coord = (-floor(sx/2):ceil(sx/2)-1);
y_coord = (-floor(sy/2):ceil(sy/2)-1);





Sc= conj(S);
      
         
        num_pixels = sy*sx;
        x_coord_mat = repmat(x_coord, [sy 1]);
        y_coord_mat = repmat(y_coord(:),[1 sx]); 
        
        
     
        
        
        kx = kx(:);
        ky = ky(:);        
        t_s1 = t_s1(:);
         t_s2 = t_s2(:);
        x_coord_mat=x_coord_mat(:);
        y_coord_mat=y_coord_mat(:);
        
       %dd=repmat([1:109]',[1 184])/109;
        dB0_rads=dB0_rads+beta;
        v = zeros(sy*sx,nch);
        for chn = 1:nch
            u_c1 = u1(:,:,chn);
            u_c1 = u_c1(:);
            u_c2 = u2(:,:,chn);
            u_c2 = u_c2(:);
            parfor pix = 1:num_pixels  %par
                % Phase due to encoding gradients and B0 inhomogeneities
                pix_phase1 = ( kx .* x_coord_mat(pix) ) + ( ky .* (y_coord_mat(pix)) ) + ( t_s1 .* 2*pi*(dB0_rads(pix)) ); % this is correct, checked against mex function
                pix_phase2 = ( kx .* x_coord_mat(pix) ) + ( ky .* (y_coord_mat(pix)) ) + ( t_s2 .* 2*pi*(dB0_rads(pix)) ); 
                
            %      pix_phase1 = -beta+( kx .* x_coord_mat(pix) ) + ( ky .* (y_coord_mat(pix)) ) + ( t_s1 .* 2*pi*(dB0_rads(pix)) ); % this is correct, checked against mex function
            %    pix_phase2 = beta+( kx .* x_coord_mat(pix) ) + ( ky .* (y_coord_mat(pix)) ) + ( t_s2 .* 2*pi*(dB0_rads(pix)) ); 
                
                % pix_phase = ( kx .* x_coord_mat(pix) ) + ( ky .* y_coord_mat(pix) ) + ( dd(:) .* dB0_rads(pix) );
                sol1 =  u_c1.' * exp( 1i.* (pix_phase1) ) ; 
                sol2 =  u_c2.' * exp( 1i.* (pix_phase2) ) ; 
               % sol= exp(-1i * pix_phase)' * u_c(:);
                
                v1(pix,chn) = sol1; %crazily doing this is faster!
                v2(pix,chn) = sol2;
            end
        end
        v1 = sum(reshape(v1,[sy sx nch]) .* Sc,3);
        v2 = sum(reshape(v2,[sy sx nch]) .* Sc,3);
       
        

      %  v1=v1./(sqrt(sum(abs(Sc).^2,3)+eps)); %original
      % v2=v2./(sqrt(sum(abs(Sc).^2,3)+eps)); %original
      
%       v1=v1./((sum(abs(Sc).^2,3)+0.1));
%       v2=v2./((sum(abs(Sc).^2,3)+0.1));
%        pixel_shift=(dB0_rads/bw_per_pixel);
% for ii=2:size(v1,1)-1;
%     for jj=1:size(v1,2);
% temp=[ pixel_shift(ii-1,jj) pixel_shift(ii,jj) pixel_shift(ii+1,jj) ];
% p=polyfit([1:3],temp,1);p=p(1);
% v1(ii,jj)=(1+p)*v1(ii,jj);
% v2(ii,jj)=(1-p)*v2(ii,jj);    
% end;end
%        
       
     %v=v./((sum(abs(Sc).^2,3)+eps));
     %   v=v./((sum(abs(Sc).^2,3)+eps));
     
    %v=v./(sum(abs(Sc).^2,3)+eps);
   
output=sum(abs(v1(:)-v2(:)).^2)./sum(abs(v1(:)).^2);
end