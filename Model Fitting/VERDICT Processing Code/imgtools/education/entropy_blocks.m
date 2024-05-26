function entropy_blocks


datadir = 'C:\Users\ucacdat\home\data\20140629_Tromso\2D_T1w' ;
fn = 'res320x320_2D_ima1.mat' ;

d = load(fullfile(datadir,fn)) ;
[ny nx] = size(d.dat) ;


k1 = i2k(d.dat) ;
kunits = [] ; dim = 1 ;
kmotion = k1 ;


figure('Units','pixels','Position',[100 100 600 400],'Name','k-space power')
hp = plot( sum((k1.*conj(k1)),2), [1:ny]-DCgp(d.dat,1));
set(hp,'LineWidth',2)

eshow(k1)



delta_disp = [3 0 0] ;
planes = [1:floor(ny/2)] ;
k_out = d_apply(k1, kunits, delta_disp, dim, planes) ;
kmotion(planes,:) = k_out ;

delta_disp = [0 2 0] ;
planes = [floor(ny/2): ny] ;
k_out = d_apply(k1, kunits, delta_disp, dim, planes) ;
kmotion(planes,:) = k_out ;

iout = k2i(kmotion) ;
eshow(iout)
eshow(d.dat)

disp(['Energy in:',num2str(sum(sum(d.dat.*conj(d.dat)))),...
    '  out:',num2str(sum(sum(iout.*conj(iout))))])

N0 = sqrt(sum(sum(d.dat.*conj(d.dat)))) ;

disp(['Entropy in:',num2str(entrp(d.dat,N0)),'  out:',num2str(entrp(iout,N0))])

% fent = @(x) fnhe(x,N0) ;
% nbd = [40 40] ;
% eout = nlfilter(iout,nbd,fent) ;
% ein =  nlfilter(d.dat,nbd,fent) ;
% eshow(eout-ein)

% Gradient Entropy
[FX,FY] = gradient(d.dat);
[FXout, FYout] = gradient(iout) ;
gout = abs(FYout)+abs(FXout) ;
gin = abs(FY) +abs(FX);
eshow(gout)
eshow(gin)
Ng0 = sum(gout(:));
fgent = @(x) fble(x,Ng0) ;
nbd = [20 20] ;
egout = blockproc(gout,nbd,fgent) ;
egin =  blockproc(gin,nbd,fgent) ;
de = (egout-egin) ;

% Normal Entropy
% fent = @(x) fble(x,N0) ;
% nbd = [20 20] ;
% eout = blockproc(iout,nbd,fent) ;
% ein =  blockproc(d.dat,nbd,fent) ;
% de = (eout-ein) ;

nc = 256 ;
map = cat(2, linspace(0,1,nc)' , zeros([nc 1]), linspace(1,0,nc)' ) ;
figure
mxde = max(de(:)) ;
mnde = min(de(:)) ;
rng = max(mxde,mnde)*2;
imshow((de+rng/2)*(255/rng),map)


% CF plot
fac = linspace(0,1,100);
ent = zeros(size(fac)) ;
gent = zeros(size(fac)) ;

for ifac = 1:length(fac)
    kmotion = k1 ;
    delta_disp = fac(ifac)*[3 0 0] ;
    planes = [1:floor(ny/2)] ;
    k_out = d_apply(k1, kunits, delta_disp, dim, planes) ;
    kmotion(planes,:) = k_out ;

    delta_disp = fac(ifac)*[0 2 0] ;
    planes = [floor(ny/2): ny] ;
    k_out = d_apply(k1, kunits, delta_disp, dim, planes) ;
    kmotion(planes,:) = k_out ;

    iout = k2i(kmotion) ;
    
    N0 = sum(sqrt(iout(:).*conj(iout(:))));
    ent(ifac) = entrp(iout,N0) ;
    [FXout, FYout] = gradient(iout) ;
    gout = abs(FYout)+abs(FXout) ;
    Ng0 = sum(gout(:)) ;
    gent(ifac) = entrp(gout,Ng0) ;
end
ent = (ent-min(ent))/(max(ent)-min(ent)) ;
gent = (gent-min(gent))/(max(gent)-min(gent)) ;
figure('Units','pixels','Position',[100 100 600 400],'Name','Entropies')
plot(fac,ent,'LineWidth',2), hold on, plot(fac,gent,'LineWidth',2)
xlabel('Displacement')
ylabel('Cost Function')
legend('Entropy','Gradient entropy')



function ent = entrp(x,N0)
x = abs(x)/N0 ;
lo = find(x < realmin) ;
x(lo) = ones(size(lo)) ;

ent = sum(-x(:).*log(x(:))) ;

function nhent = fnhe(x,N0)
% Entropy neighborhood
x = abs(x)/N0 ;
lo = find(x < realmin) ;
x(lo) = ones(size(lo)) ;

nhent = sum(-x(:).*log(x(:))) ;

function blent = fble(bs,N0)
% Entropy block
x = abs(bs.data)/N0 ;
lo = find(x < realmin) ;
x(lo) = ones(size(lo)) ;

blent = sum(-x(:).*log(x(:))) ;
blent = ones(size(x))*blent ;
