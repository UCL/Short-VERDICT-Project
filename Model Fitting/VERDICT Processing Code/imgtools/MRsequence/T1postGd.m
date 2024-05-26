function T1postGd
% T1postGd Modelling of T1W images post Gadolinium injection
% Aim to maximise visibility of uptake.

% T1s in ms
% From: J Magn Reson Imaging. 1999 Apr;9(4):531-8.
WM_T1 = 832;  % White matter at 3T (832ms at lower end of literature)

GM_T1 = 1200; % Grey matter T1 at 3T

GD_T1s = [100 150] ;
fat_T1 = 370 ;


thresh = 0.02 ; % threshold for weak refernce signal (no ratio calc)

T1s = [GM_T1 GD_T1s] ;

% Flip angles
FAs = [2:2:40] ;
% TRs = [2 4 6 8 10 12 14 16 18 20 25 30] ;
TRs = [  6 8 10 15 20] ;

nfa = length(FAs); ntr = length(TRs) ; nt1 = length(T1s);
Sref = zeros([nfa ntr]);
ST1 = zeros([nfa ntr nt1]) ;

for ifa = 1:nfa
    for itr = 1:ntr
       Sref(ifa,itr) = ernstfn(WM_T1, FAs(ifa), TRs(itr)) ; 
       Ss = ernstfn(T1s, FAs(ifa), TRs(itr)) ; 
       ST1(ifa,itr,:) = reshape(Ss,[1 1 nt1]) ;
    end
end

loc_low = find(repmat(Sref,[1 1 nt1]) < thresh) ;

Srat = ST1./repmat(Sref,[1 1 nt1]) ;
Srat(loc_low) = 0 ;

hf1 = figure ;
hold on
for it1 = 1:nt1
    %mesh(TRs, FAs, Srat(:,:,it1))
    mesh(TRs, FAs, ST1(:,:,it1)-Sref)
end
%xlabel('TR'), ylabel('FA'), zlabel('Srat')
xlabel('TR'), ylabel('FA'), zlabel('Sdiff')
grid on


hf2 = figure;
plot(FAs,Sref), grid on
legend(num2str(TRs')), xlabel('FA')
hold on
plot(FAs, squeeze(ST1(:,3,:)))

% Ernst curve for each T1 and TR
for itr = 1:ntr
    
    for ifa = 1:nfa
       S_WM(ifa,itr) = ernstfn(WM_T1, FAs(ifa), TRs(itr)) ;
       S_GM(ifa,itr) = ernstfn(GM_T1, FAs(ifa), TRs(itr) );
       S_fat(ifa,itr) = ernstfn(fat_T1, FAs(ifa), TRs(itr) );
       S_T150(ifa,itr) = ernstfn(150, FAs(ifa), TRs(itr) );
       S_T100(ifa,itr) = ernstfn(100, FAs(ifa), TRs(itr) );
    end
    hf3 = figure('Name',['TR: ',num2str(TRs(itr))]) ;
    plot(FAs,S_GM(:,itr),'g'), hold on, grid on
    xlabel('FA'),ylabel('Signal'), title(['TR: ',num2str(TRs(itr))])
    axis([0 40 0 0.35])
    plot(FAs,S_WM(:,itr),'k')
    plot(FAs,S_fat(:,itr),'r')
    plot(FAs,S_T150(:,itr),'b')
    plot(FAs,S_T100(:,itr),'b--')
    legend('GM','WM','fat','T1 150','T1 100')
end


