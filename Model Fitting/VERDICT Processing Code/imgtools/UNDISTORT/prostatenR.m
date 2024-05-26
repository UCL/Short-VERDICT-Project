function prostatenR
% PROSTATENR
%  Investigation into prostate B0 correction using multiple R and/or phase
%  encode timings
%
% Problems we aim to solve or understand
%    Unfolding pile up ("indeterminate") extent.
%    Fluctuations in B0 affecting reconstruction
%     select a trajectory that is either robust to B0 change, or,
%       allows the change to be measured.
%    Model the importance of missing B0 data in the rectum
%    Enable determination of the otion-induced phase
%    Discretisation effects
%   
% Up and Down offer a big difference and thus potentially "orthogonal"
% measures that should help wth the unfolding. Their disadvantages are:
%    other sequence attributes reverse, e.g. readout gradient, diffusion
%    gradients and thus eddy currents. Coordinate shifts.
%    Even though N>2, only up and down.
%    If a registration or comparison step is needed, the images are very
%    different in terms of distortions.
% Up and Down are just +1 and -1 time step. Other time steps are possible,
% either by a deliberate manipuation of gradients, or use of different
% (probably fractional) parallel imsging factors.
%
% Some options will change the echo time which may be beneficial if also
% doing T2 modeling, but may complicate finding a solution.
%
%
%
% To add: 
%   B0 that varies with meas- would have a static, fluctuating and correction component.
%
%   higher b-value prostate with more noise
%
% David Atkinson  D.Atkinson@ucl.ac.uk
%
% See also sysmatv prostate1d gibbs_explain 

 n = 168 ; col = 60 ; % use column 80 when n=168, 60 when n=128
%n = 128 ; col = 60 ; % use column 80 when n=168, 60 when n=128
img = prostate(n) ;

% profiles through prostate and rectum
xin = img(:,col) ; % use column 80 when n=168, 60 when n=128

% Filter (mild) to reduce ringing issues
xin = imgaussfilt(xin) ;

% Noise to add later
% 30 is bradly similar to b=1000 after averaging
% 60 for b=0 after averaging

snr = 300   
signal = 0.7 ; % prostate phantom

opt.verbose = true ;
opt.scale = 'balanced'; opt.vec = true ;
opt.FOV = 220 ;
opt.Rreq =  [1.5 1.5 ] ;  % undersampling factors requested 
opt.pFskip = [1 1] ;
opt.tstep = [1 -1] * 0.98e-3 ; % corresponding time steps (s)
opt.partialFReq = 1 ;
precond = false ; % seems to make solution a bit worse (as implemented).
opt.precond = precond ;
maxit = 10 ;
opt.maxit = maxit ;

% Coil sensitivities
csens = ones([n 2]) ;
csens(:,1) = exp(-([1:n]'- 0).^2/n/30) ;
csens(:,2) = exp(-([1:n]'- n).^2/n/30) ;
opt.csens = csens ;

% motion induced phases
% % mp(:,1) = exp(2i*pi * (1-exp(-([1:n]-20).^2/n/60))) ;
% % mp(:,2) = exp(2i*pi * exp(-([1:n]-100).^2/n/70)) ;
% mp(:,3) = exp(2i*pi * (1-exp(-([1:n]-70).^2/n/60))) ;
% mp(:,4) = exp(2i*pi * exp(-([1:n]-40).^2/n/70)) ;
% % opt.mp = mp ;
    

% The B0 shows a dip near the prostate/rectum edge
% b0fun takes a spatial postion as input

opt.b0fun       = @(ir)  b0rectum(ir, 'irectA',110,'irectP',127) ;
opt.b0funmodify = @(ir)  b0rectum(ir, 'offset',10,'shift',2,'irectA',110, ...
    'irectP',127, 'low_signal_loc', [110:127]) ;
    
% opt.b0fun       = @(r) 0
% opt.b0funmodify = @(r) 0

[~,opt] = sysmatv(xin,'parse', opt) ;  % parse to set other opt fields

% set time function. Takes itraj (unused) and index of k-space coord as
% inputs. This index is converted to time by ckt (set in sysmatv)
ckt = opt.ckt ;
opt.Tfun  = @(itraj, ik) (ckt(ik))  ;

% calc k-space yout
[yout, optout] = sysmatv(xin,'notransp', opt) ;

% Add noise. Use noise variance determined from image signal and expected
% SNR. Although yout may contain multiple measures, the noise added should
% be correct.

youtnonoise = yout ;
rng('default')
yout = addnoise(yout,'snr',snr,'signal',signal,'indomain','kspace','scale','balanced') ;

csensplot(optout, xin) % plot the coil sensitivites
b0plot(optout, xin) % plot the B0 field
tplot(optout)  % plot k-space line timings and partial Fourier excluded
ftyplot(yout, optout, xin) % plot FFT of k-space (no B0 or SENSE corrections)

% Estimate motion induced phase (and f0 drift)
newmp = mpestimate(yout, opt) ;
opt.mp = newmp ;    

% Do LSQR on k-space 
opt.verbose = false ;
fmv = @(x,flag)sysmatv(x, flag, opt) ;
mfun = @(x, flag)msysmatv(x, flag, opt) ;

if precond
    [x1, flag,relres,iter,resvec] = lsqr(fmv, yout, [], maxit, mfun ) ;
else
    [x1, flag,relres,iter,resvec] = lsqr(fmv, yout, [], maxit) ;
end

figure('Name','prostatenR')
plot(opt.cr1, abs(x1), 'DisplayName','x1 from LSQR'), hold on, grid on
plot(opt.cr1, xin, 'DisplayName','xin truth')

xlabel('Position (mm)')
set(findobj(gcf,'Type','line'),'LineWidth',2)
title(['R ',num2str(opt.Rreq),'.  tstep: ',num2str(opt.tstep*1000), ...
    ' pF: ',num2str(opt.partialFAct)])
legend

% recon without B0
opt.b0fun = @(r) 0 ;
fmv = @(x,flag)sysmatv(x, flag, opt) ;
mfun = @(x, flag)msysmatv(x, flag, opt) ;

if precond
    [xnob0, flag,relres,iter,resvec] = lsqr(fmv, yout, [], maxit, mfun) ;
    [xnob0nonoise, flag,relres,iter,resvec] = lsqr(fmv, youtnonoise, [], maxit, mfun) ;
else
    [xnob0, flag,relres,iter,resvec] = lsqr(fmv, yout, [], maxit) ;
    [xnob0nonoise, flag,relres,iter,resvec] = lsqr(fmv, youtnonoise, [], maxit) ;
end
    
figure('Name','no B0')
plot(opt.cr1, abs(xnob0), 'DisplayName','x1 no B0 from LSQR'), hold on, grid on
plot(opt.cr1, abs(xnob0nonoise), 'DisplayName','x1 no B0 no noise from LSQR')
plot(opt.cr1, xin, 'DisplayName','xin truth')

xlabel('Position (mm)')
set(findobj(gcf,'Type','line'),'LineWidth',2)
title('')
legend

% recon with modified B0
opt.b0fun = opt.b0funmodify ;

fmv = @(x,flag)sysmatv(x, flag, opt) ;
mfun = @(x, flag)msysmatv(x, flag, opt) ;
if precond
    [xb0m, flag,relres,iter,resvec] = lsqr(fmv, yout, [], maxit, mfun) ;
else
    [xb0m, flag,relres,iter,resvec] = lsqr(fmv, yout, [], maxit) ;
end

figure('Name','B0 modify')
plot(opt.cr1, abs(xb0m), 'DisplayName','x1 B0 modify'), hold on, grid on
plot(opt.cr1, xin, 'DisplayName','xin truth')

xlabel('Position (mm)')
set(findobj(gcf,'Type','line'),'LineWidth',2)
title(['R ',num2str(opt.Rreq),'.  tstep: ',num2str(opt.tstep*1000), ...
    ' pF: ',num2str(opt.partialFAct)])
legend

end


% ---


% ---
function csensplot(opt, xin)
% CSENSPLOT Plot coil sensitivities and image if input

csens = opt.csens ;

[~,nc] = size(csens) ;
figure('Name','Coil Sensitivities')
for ic = 1: nc
    plot(opt.cr1, csens(:,ic)), hold on, grid on
end
plot(opt.cr1, sqrt(sum( csens.*conj(csens) ,2)) ,'r--')
plot(opt.cr1,xin,'k')
end

%---
function tplot(opt)
% TPLOT Plot timings of acquisitions of k-space lines

nck = length(opt.ck1) ; % number of k-space coordinates (these concatenate for nR>1)
indck = [1:nck] ;       % indexing of these
loc0 = find(opt.kmask == 0) ;
figure('Name','tplot')
plot(indck,opt.ck1*opt.FOV, 'DisplayName', ' k-space coord*FOV'), hold on, grid on
ts = opt.Tfun(1,[1:nck]) ;
plot(indck,ts*1000, 'DisplayName', ' t (ms)')
if ~isempty(loc0), plot(indck(loc0),0,'ro',  'DisplayName', ' masked to zero'), end
legend
xlabel('index of k-space coordinate')
ylabel('k-space coord*FOV or Time (ms)')

figure('Name','tvsck')
nR = size(opt.Rreq,2) ;
for iR = 1:nR
   ckindthisR = opt.ckstart(iR):opt.ckend(iR) ;
   ck1thisR = opt.ck1(ckindthisR) ;
   cktthisR = opt.ckt(ckindthisR) ;
   plot(ck1thisR*opt.FOV, 1000*cktthisR ), hold on
   % indicate which are masked with markers
   loc0 = find(opt.kmask(ckindthisR) == 0) ;
   plot(ck1thisR(loc0)*opt.FOV, 1000*cktthisR(loc0),'ro')
end
grid on
xlabel('k-space coord*FOV')
ylabel('Time (ms)')

    
end

% ---
function ftyplot(yout, opt, xin) 
figure('Name', 'ftyplot')
yout = reshape(yout, opt.szyf) ;
for iR = 1: length(opt.Ract)
    xtot = 0 ;  % running total for sum of squares of signal
    for icoil = 1: opt.ncoil
        xR = k2i(yout(opt.ckstart(iR):opt.ckend(iR), 1, icoil)) ;
        % try to get on same scales to help with comparisons
        xlocs = ( [1:length(xR)]-sz2DC(length(xR) ) ) .* opt.FOV/length(xR)/opt.Ract(iR) ;
        plot(xlocs,abs(xR),'DisplayName', ['abs(xr), R ',num2str(opt.Ract(iR))])
        hold on
        xtot = xtot + xR.*conj(xR) ;
    end   
    plot(xlocs,sqrt(xtot), 'DisplayName','sqrt sos img')
end
legend, grid on
set(findobj(gcf,'Type','line'),'LineWidth',2)
end

    
    