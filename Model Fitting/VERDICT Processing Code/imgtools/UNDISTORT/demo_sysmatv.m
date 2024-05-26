function demo_sysmatv
% DEMO_SYSMATV Demos and tests sysmatv
% Shows use as Fourier Transform and function for LSQR
% Currently 1D in spatial dimensions.
%
%
% To do:
%  Split into test suite
%  
% David Atkinson . D.Atkinson@ucl.ac.uk
% See also SYSMATV


% Test fwd followd by adj gives same for B0=0
% Compare with k2i, i2k .

D = load('mri') ;
xin = double(D.D(65,:,1,10)) ; % use line through mri slice
xin = xin(:) ;

opt.verbose = true ;
opt.scale = 'i2k' ;
opt.partialFReq = 0.7 ;

[yout, optout] = sysmatv(xin, 'notransp', opt) ;

% Plot k-space and compare with i2k
figure('Name','kspace')
plot(optout.ck1, abs(yout), 'DisplayName','abs fwd'), hold on
plot(optout.ck1, abs(i2k(xin)), 'DisplayName','abs i2k')
yyaxis right
plot(optout.ck1, angle(yout), 'DisplayName','phase fwd'), hold on
plot(optout.ck1, angle(i2k(xin)), 'DisplayName','phase i2k')

legend


% Transform back to image domain (here adjoint is inverse as B0=0)
[newim] = sysmatv(yout, 'transp', optout) ;

figure('Name','img')
plot(optout.cr1, xin, 'DisplayName', 'xin'), hold on
plot(optout.cr1, abs(newim), 'DisplayName','newim')

yyaxis right
plot(optout.cr1, angle(xin), 'DisplayName', 'angle xin'), hold on
plot(optout.cr1, angle(newim), 'DisplayName','angle newim')
yyaxis left
legend


% --- Demo function handle
% The complex transpose is only equivalent to the inverse FT for no B0.
opt.scale = 'i2k';
opt.partialFReq = 1 ;

ftfwd = @(dat) sysmatv(dat,'notransp', opt) ;
ftadj = @(dat) sysmatv(dat,'transp', opt) ;

kin = ftfwd(xin) ; % Can use similar to k2i 
ftk = ftadj(kin) ; % and i2k

figure('Name','fhandle demo')
plot(abs(xin), 'DisplayName','abs xin'), hold on, legend
plot(abs(ftk), 'DisplayName','abs ftk')
yyaxis right
plot(angle(xin), 'DisplayName','angle xin'), hold on
plot(angle(ftk), 'DisplayName','angle ftk')
yyaxis left

% --- Check constant B0 offset gives the expected shift
% find location of DC line
kdc = find(abs(optout.ck1) < 0.001);
if length(kdc) ~= 1; warning(['DC not found']); end

opt.b0fun = @(r) 10 /1e-3 / length(xin) ;  % should be 10 pixels
opt.Tfun  = @(itraj, k) (k-kdc)*1e-3 ;

[yout, optout] = sysmatv(xin, 'notransp', opt) ;
newim = k2i(yout) ;
figure('Name','B0 shift check')
plot(optout.cr1,xin, 'DisplayName','xin')
hold on, grid on
plot(optout.cr1, abs(newim), 'DisplayName', 'newim')
plot(optout.cr1, angle(newim), 'DisplayName', 'angle newim')
legend



% --- Demonstrate Gibb's ringing by calculating larger k-space region first.
% FOV remains the same, k-space samples for low res taken from central part
% of higher res
opthr.scale = 'step' ;

[youthr, opthrout] = sysmatv(xin,'notransp', opthr) ;

% take central portion and k2i using sysmatv
orig = find(abs(opthrout.cr1) < 0.001); % location of image origin
cent = orig-32:orig+31 ; % central indices for k-space
optlr.scale = 'step';
optlr.ck1 = opthrout.ck1(cent) ; % central k-space coordinates from hr
optlr.FOV = opthrout.FOV ;    % FOV remains the same
[imlr, optlrout] = sysmatv(youthr(cent), 'transp', optlr) ;

figure('Name','LR from HR')
plot(optlrout.cr1, abs(imlr), 'DisplayName','abs imlr') , hold on
plot(opthrout.cr1, xin, 'DisplayName','xin')
plot(optlrout.cr1, angle(imlr), 'DisplayName','angle imlr') , hold on
legend


% ---- Check conjugate phase recon 
% 
opt.b0fun = @(r) 10 / (1 + exp(-r/4)) ;
opt.Tfun  = @(itraj, k) (k-sz2DC(xin,1))*1e-3 ;

[yout, optout] = sysmatv(xin, 'notransp', opt) ;

[xcp] = sysmatv(yout, 'transp', optout) ; % conjugate phase

optk2i = optout;
optk2i.b0fun = @(r) 0 ;

[xk2i] = sysmatv(yout, 'transp', optk2i) ;

figure('Name','conj phase')
plot(optout.cr1, xin, 'DisplayName','xin'), hold on, grid on, legend
plot(optout.cr1, abs(xk2i), 'DisplayName','abs xk2i')
plot(optout.cr1, abs(xcp), 'DisplayName','abs xcp')
yyaxis right
plot(optout.cr1, angle(xk2i), 'DisplayName','angle xk2i')
plot(optout.cr1, angle(xcp), 'DisplayName','angle xcp')


% ---- Check use in LSQR
% Use 'balanced' scale and vec output
x0 = zeros(size(xin)) ;

opt.b0fun = @(r) 10 / (1 + exp(-r/4)) ;
opt.Tfun  = @(itraj, k) (k-sz2DC(xin,1))*1e-3 ;
opt.verbose = false ; % Need these for LSQR
opt.scale = 'balanced' ;
opt.vec = true ;

fmv = @(x,flag)sysmatv(x, flag, opt) ;

[yout, optout] = sysmatv(xin, 'notransp', opt) ;

[x1, flag,relres,iter,resvec] = lsqr(fmv, yout, [], [], [],[],x0) ;

optk2i = opt ; optk2i.b0fun = @(r) 0 ;  % Just a k2i with no B0
[xk2i] = sysmatv(yout, 'transp', optk2i) ;
figure('Name','LSQR')
plot(optout.cr1, xin, 'DisplayName','xin'), hold on
legend, grid on
plot(optout.cr1, abs(x1), 'DisplayName','abs x1')
plot(optout.cr1, abs(xk2i), 'DisplayName','abs xk2i')

yyaxis right
plot(optout.cr1, angle(x1), 'DisplayName','angle x1')
yyaxis left

% --- Multiple coils, no undersampling.
% Can't use sysmatv as a FT routine for more than one coil
N = length(xin) ;
csens = ones([N 2]) ;
csens(:,1) = exp(-([1:N]- 30).^2/N) ;
csens(:,2) = exp(-([1:N]- 90).^2/N) ;

optc.csens = csens ;
optc.scale = 'balanced' ; optc.vec = true ;


[yout, optcout] = sysmatv(xin, 'notransp', optc) ;

fmv = @(x,flag)sysmatv(x, flag, optcout) ;

[x1, flag,relres,iter,resvec] = lsqr(fmv, yout, [], [], [],[],x0) ;


figure('Name','C fully sampled')
plot(optcout.cr1, xin, 'DisplayName', 'xin'), hold on, grid on
plot(optcout.cr1, xin.*csens(:,1), 'DisplayName', 'ic1')
plot(optcout.cr1, xin.*csens(:,2), 'DisplayName', 'ic2')
plot(optcout.cr1, abs(x1), 'DisplayName', 'x1')
legend

% --- undersampled CG
% 
N = length(xin) ;
csens = ones([N 2]) ;
csens(:,1) = exp(-([1:N]- 30).^2/N/5) ;
csens(:,2) = exp(-([1:N]- 90).^2/N/5) ;

optc.csens = csens ;
optc.scale = 'balanced' ; optc.vec = true ;
optc.Rreq = 2 ;

[yout, optcout] = sysmatv(xin, 'notransp', optc) ;

fmv = @(x,flag)sysmatv(x, flag, optcout) ;

[x1, flag,relres,iter,resvec] = lsqr(fmv, yout, [], [], [],[],x0) ;


figure('Name','C under sampled')
plot(optcout.cr1, xin, 'DisplayName', 'xin'), hold on, grid on
plot(optcout.cr1, xin.*csens(:,1), 'DisplayName', 'ic1')
plot(optcout.cr1, xin.*csens(:,2), 'DisplayName', 'ic2')
plot(optcout.cr1, abs(x1), 'DisplayName', 'x1')
legend

% --- undersampled CG with partialF
% 
N = length(xin) ;
csens = ones([N 2]) ;
csens(:,1) = exp(-([1:N]- 30).^2/N/5) ;
csens(:,2) = exp(-([1:N]- 90).^2/N/5) ;

optc = [] ;
optc.partialFReq = 0.7 ;
optc.csens = csens ;
optc.scale = 'balanced' ; optc.vec = true ;
optc.Rreq = 1.7 ;

[yout, optcout] = sysmatv(xin, 'notransp', optc) ;

fmv = @(x,flag)sysmatv(x, flag, optcout) ;

[x1, flag,relres,iter,resvec] = lsqr(fmv, yout, [], [], [],[],x0) ;


figure('Name','C under sampled and pF')
plot(optcout.cr1, xin, 'DisplayName', 'xin'), hold on, grid on
plot(optcout.cr1, xin.*csens(:,1), 'DisplayName', 'ic1')
plot(optcout.cr1, xin.*csens(:,2), 'DisplayName', 'ic2')
plot(optcout.cr1, abs(x1), 'DisplayName', 'x1')
legend
