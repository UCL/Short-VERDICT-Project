function [v] = Eh_v5_motionmatrixoffset(u,arg1,arg2)


Sk1=Eh_v7_motionmatrix(u(1:end/2,:,:),arg1);
Sk2=Eh_v7_motionmatrix(u(end/2+1:end,:,:),arg2);



%  pixel_shift=(arg1.dB0_rads/(2*pi))/arg1.bw_per_pixel;
% for ii=3:size(Sk1,1)-2;
%     for jj=1:size(Sk1,2);
% temp=[pixel_shift(ii-2,jj) pixel_shift(ii-1,jj) pixel_shift(ii,jj) pixel_shift(ii+1,jj) pixel_shift(ii+2,jj)];
% p=polyfit([1:5],temp,1);p=p(1);
% Sk1(ii,jj)=(1+p)*Sk1(ii,jj);%1+
% Sk2(ii,jj)=(1-p)*Sk2(ii,jj); %1-   
% end;end


v=(Sk1+Sk2)/2;
tic;toc %E_v3
end
