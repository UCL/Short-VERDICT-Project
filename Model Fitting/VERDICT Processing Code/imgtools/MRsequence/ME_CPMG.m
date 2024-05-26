%%%% ME_CPMG 
% Adapted from Shaihan Malik's  Test 2: Multiecho CPMG


% Set the b pool to have zero fraction

%%% Water excahnge model. For interpretation, compartment A is the larger
%%% (slow relaxing) and B is the small, fast relaxing compartment (myelin
%%% water?)
f = 0.20; %<--- pool b fraction (this is fast relaxing pool) i.e.  myelin water fraction
fka = 2e-2; % This is ka->b (i.e. Ksf in mcdespot; fb*kb = fa*ka, so kb = (1-fb)*ka/fb -- this is 1/tau
T1l = [1500 1500];
T2l = [300 300];

T1s = [1000 1000] ;
T2s = [50 50] ;


%%% Pulse sequence parameters
ESP=10;
Necho = 16 ;
a0 = d2r([90 140 100*ones(1,Necho-1)]); % add 90 + theta/2 to improve SS approach (if required)
Npulse = length(a0);
Necho = Npulse-1;




%%% Analyse the echoes using NNLS (Whittall KP, MacKay AL. Quantitative
%%% interpretation of NMR relaxation data. J. Magn. Reson. 1989;84:134?152.)
nt2 = 2001;
t2s = linspace(10,500,nt2);
r2s = 1./t2s;
tt = ESP*(1:Necho);
% S =exp(-r2s(:)*tt);  % [nt2 Necho] single compartment fits. Assumes 180s

S = zeros([nt2 Necho]) ;
for it2=1:nt2
    S(it2,:) = abs(EPGX_TSE_BM(a0,ESP,T1l,[t2s(it2) t2s(it2)],0,0,'delta',0));
end

sl = EPGX_TSE_BM(a0,ESP,T1l,T2l,0,0,'delta',0);
ss = EPGX_TSE_BM(a0,ESP,T1s,T2s,0,0,'delta',0);

lwf = 0.1 
sig = lwf*abs(sl)+ (1-lwf)*abs(ss) ;
figure
plot(tt,lwf*abs(sl)), hold on
plot(tt,(1-lwf)*abs(ss))
plot(tt,sig)

legend('sl','ss','ss+sl')


t2sol = lsqnonneg(S',abs(sig(:)));
[~,pks] = findpeaks(t2sol);
tmp = t2s(pks);
[~,i100] = min(abs(t2s-100)) ;
fapp = sum(t2sol(1:i100)) / sum(t2sol(:)); %<-- look 5 points either side
disp(['1-fapp = ',num2str(1-fapp)])
figure
plot(t2s,t2sol)

%% 1. Look at estimated T2s/fraction as a function of B1 offset and exchange rate

nk=32;ntx=32;
delta = 0;
ka = linspace(0,2.5e-3,nk);
tx = linspace(0.75,1.25,ntx); % B1 scalings

t2app = zeros(nk,ntx,2);
fapp = zeros(nk,ntx);
for ii=1:nk
    for jj=1:ntx
        s = EPGX_TSE_BM(a0*tx(jj),ESP,T1,T2,f,ka(ii),'delta',delta);
        t2sol = lsqnonneg(S',abs(s(:)));
        [~,pks] = findpeaks(t2sol);
        tmp = t2s(pks);
        t2app(ii,jj,:)=tmp(1:2);%<- in case more than 2 peaks
        % fraction
        fapp(ii,jj) = sum(t2sol(pks(1)+(-5:5))) / sum(t2sol(:)); %<-- look 5 points either side
    end
    disp([ii jj])
end




%% Now examine effect of delta and kx for B1scaling=1

nk=32;nd=32;
ka2 = linspace(0,2.5e-3,nk);
dppm = linspace(-1,1,nd);
delta = dppm*1e-6*3*42.6e3;% ppm * 3T * 42.6kHz/T

nt2_2 = 500;
t2s_2 = linspace(10,120,nt2_2);
r2s_2 = 1./t2s_2;
tt = ESP*(1:Necho);
S_2 =exp(-r2s_2(:)*tt);


t2app_2 = zeros(nk,nd,2);
fapp_2 = zeros(nk,nd);
for ii=1:nk
    for jj=1:nd
        s = EPGX_TSE_BM(a0,ESP,T1,T2,f,ka2(ii),'delta',delta(jj));
        t2sol = lsqnonneg(S_2',abs(s(:)));
        [~,pks] = findpeaks(t2sol);
        tmp = t2s_2(pks);
        t2app_2(ii,jj,:)=tmp(1:2);%<- in case more than 2 peaks
        % fraction
        fapp_2(ii,jj) = sum(t2sol(pks(1)+(-5:5))) / sum(t2sol(:)); %<-- look 5 points either side
    end
    disp([ii jj])
end



%% Combined figure

%%% example curve
[s,Fn] = EPGX_TSE_BM(a0*1.1,ESP,T1,T2,f,2e-3);

t2sol = lsqnonneg(S',abs(s(:)));
[~,pks] = findpeaks(t2sol);
disp('t2s(pks):')
t2s(pks)
sum(t2sol(pks(1)+(-5:5)))/ sum(t2sol(:)) %<-- look 5 points either side

figure(1);clf;
nr=2;nc=3;

subplot(nr,nc,2)
imagesc(tx,ka*1e3,fapp,[0.12 0.22])
hold
[cc,h]=contour(tx,ka*1e3,fapp,0.1:0.02:0.24,'linewidth',1.5);
h.LineColor = [1 1 1];
clabel(cc,h,'Color',[1 1 1],'fontweight','bold')

title('$$\hat{f}$$','Interpreter','latex','fontsize',16)
xlabel('B_1 scaling factor')
ylabel('k_a, s^{-1}')
set(gca,'FontSize',12)

colorbar

subplot(nr,nc,5)
imagesc(tx,ka*1e3,t2app(:,:,1),[16 45])
hold
[cc,h]=contour(tx,ka*1e3,t2app(:,:,1),16:4:36,'linewidth',1.5);
h.LineColor = [1 1 1];
clabel(cc,h,'Color',[1 1 1],'fontweight','bold')

title('$$\hat{T}_{2,b}$$','Interpreter','latex','fontsize',16)

xlabel('B_1 scaling factor')
ylabel('k_a, s^{-1}')
set(gca,'FontSize',12)

colorbar

subplot(nr,nc,4)
plot(t2s,t2sol)
grid on
hold
xlabel('T_2 / ms')
ylabel('au')
title('T_2 spectrum from NNLS')
xlim([15 105])
set(gca,'FontSize',12)

subplot(nr,nc,1)
plot((1:50)*ESP,abs(s),'linewidth',2)
grid on
hold
plot((1:50)*ESP,squeeze(abs(Fn(52,:,:))),'linewidth',2)
xlabel('Echo time / ms')
ylabel('Signal (F_0)')
title('Echo amplitudes')
xlim([1 50]*ESP)
set(gca,'FontSize',12)

subplot(nr,nc,3)
imagesc(dppm,ka*1e3,fapp_2,[0.12 0.22])
hold
[cc,h]=contour(dppm,ka*1e3,fapp_2,0.08:0.02:0.24,'linewidth',1.5);
h.LineColor = [1 1 1];
clabel(cc,h,'Color',[1 1 1],'fontweight','bold')

title('$$\hat{f}$$','Interpreter','latex','fontsize',16)
xlabel('\delta_b ppm (@3T)')
ylabel('k_a, s^{-1}')
set(gca,'FontSize',12)

colorbar

subplot(nr,nc,6)
imagesc(dppm,ka*1e3,t2app_2(:,:,1),[10 22])
hold
[cc,h]=contour(dppm,ka*1e3,t2app_2(:,:,1),15:25,'linewidth',1.5);
h.LineColor = [1 1 1];
clabel(cc,h,'Color',[1 1 1],'fontweight','bold')

title('$$\hat{T}_{2,b}$$','Interpreter','latex')
xlabel('\delta_b ppm (@3T)')
ylabel('k_a, s^{-1}')
set(gca,'FontSize',12)

colorbar

gg=get(gcf,'children');

%%% INset plot
%
axes(gg(2))
ax=axes('Position',[0.12 0.2229 0.0949 0.0996]);
box on
pp = patch(t2s([-5:5]+pks(1)),t2sol([-5:5]+pks(1)),ones([1 11]));
xlim([19 21])
grid on
pp.FaceColor = [0.75 0.75 0.];
pp.EdgeColor = [0 0.5 0.75];
aa=annotation('arrow',[0.11 0.09],[0.2104 0.1653]);
colormap viridis %<-- https://uk.mathworks.com/matlabcentral/fileexchange/51986-perceptually-uniform-colormaps
pause(0.01)

%
cbww=0.01;cbhh=0.2;
ww = 0.225;hh=0.32;
lg = 0.07;
ga = 0.09;
gg(1).Position = [lg+2*ga+3*ww+0.01    0.15    cbww    cbhh];
gg(2).Position = [lg+2*ga+2*ww    0.1    ww    hh];
gg(3).Position = [lg+2*ga+3*ww+0.01    0.65    cbww    cbhh];
gg(4).Position = [lg+2*ga+2*ww    0.6    ww    hh];
gg(7).Position = [lg+lg+2*ww+0.01    0.15    cbww    cbhh];
gg(8).Position = [lg+lg+ww    0.1    ww    hh];
gg(9).Position = [lg+lg+2*ww+0.01    0.65    cbww    cbhh];
gg(10).Position = [lg+lg+ww    0.6    ww hh];
gg(5).Position = [lg    0.6    ww    hh];
gg(6).Position = [lg    0.1    ww    hh];

%%% labels
axes(gg(5))
fs=18;
text(-30,-0.15,'(a)','fontsize',fs,'fontweight','bold')
text(-30,-1.7,'(b)','fontsize',fs,'fontweight','bold')
text(280,-0.15,'(c)','fontsize',fs,'fontweight','bold')
text(280,-1.7,'(d)','fontsize',fs,'fontweight','bold')
text(640,-0.15,'(e)','fontsize',fs,'fontweight','bold')
text(640,-1.7,'(f)','fontsize',fs,'fontweight','bold')

text(600,0.05,'$$\hat{f}$$','Interpreter','latex','fontsize',14);%,'fontangle','italic')
text(970,0.05,'$$\hat{f}$$','Interpreter','latex','fontsize',14);%,'fontangle','italic')

text(580,-1.5,'ms','fontsize',14);%,'fontangle','italic')
text(940,-1.5,'ms','fontsize',14);%,'fontangle','italic')

% legend
axes(gg(5))
legend('Total echo amplitude','Compartment a','Compartment b')

setpospap([360   231   900   450])
%
% print -r300 -dpng bin/Test2_fig1.png