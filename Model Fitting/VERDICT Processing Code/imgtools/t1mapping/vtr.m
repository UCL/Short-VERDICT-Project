function vtr
% TRs=[1:1:15]
% fa = [2 3 4 5 10 15 20] ;
% T1=1500
% 
% 
% S = zeros([length(TRs) length(fa)]) ;
% ilstr = 1;
% for ifa = 1:length(fa)
%     for itr = 1:length(TRs)
%         S(itr,ifa) = ernstfn(T1, fa(ifa), TRs(itr)) ;
%         
%     end
%     lstr{ilstr} = [num2str(fa(ifa))] ;
%     ilstr = ilstr + 1;
% end

% figure
% plot(TRs,S)
% xlabel('TR'), ylabel('S')
% legend(lstr)
% grid on

%--------------

FA = [   1.5  4   6   1.5  4  6 ] ;
TR = [    4   4   4    10  10  10] ;
M0_true = 30 ;
T1_true = 1130 
B1_true = 0.8



nm = length(TR);
for inm = 1:nm
    M(inm) = M0_true * ernstfn(T1_true,B1_true*FA(inm),TR(inm)) ;
end

fitT1 = MFAT1(M,FA,TR,[],'x0_meth','guess1','B1unknown',true) 

% T1s = [600:50:2000];
% nt1 = length(T1s) ;
% 
% for it1 = 1:nt1
%     
% B1 = [0.5:0.01:1.5];
% M0 = [0.5:0.01:1.5]; 
% 
% FAs_rad = FA / 360 * 2 * pi ;
% 
% 
% for ib1 = 1:length(B1)
%     for im0 = 1:length(M0)
%         for inm = 1:nm 
%             E = exp(-TR(inm)/T1s(it1)) ;
%             SI(inm) = M0(im0)*sin(B1(ib1)*FAs_rad(inm)).*((1-E)./(1-cos(B1(ib1)*FAs_rad(inm)).*E)) ;
%         end
%         %res = norm(squeeze(M(r,c,s,:))' - SI) ;
%         cf(ib1,im0) = norm(SI-M) ;
%     end
% end
% 
% cft(it1) = min(cf(:)) ;
% 
% end
% 
% figure
% plot(T1s,cft)
% contour(M0,B1,cf,60)
% xlabel('M0')
% ylabel('B1')
% grid
% hold on
