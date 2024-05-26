function prostate1d
% PROSTATE1D
% Aim was to investigate the observation that sometimes stripey artefacts
% appear during distortion correction of real data. When the same B0 field
% is applied in simulation, the data is correctable.  This can lead to the
% impression that the B0 map was not correct for the real data.
%
% In this demo we see:
%  Ringing can appear in simulations when phase changes due to the B0 field
%   result in non-integer pixel shifts.
%
% In the LSQR solution here, quite large oscillations appear in the solution.
% Note it is quite difficult to use a computer to simualte the acquired 
% sampled k-space because of discretisation, sampling and aliasing issues.
%
% See also gibs_explain sysmatv


nhr = 1024 ;
nlr = 128 ;
imghr = prostate(nhr) ;
imglr = prostate(nlr) ;

xinhr = imghr(:,477) ; % profiles through prostate and rectum
xinlr = imglr(:,60) ;

opt.verbose = true ;
opt.scale = 'i2k'; opt.vec = true ;
opt.FOV = nlr ;

% The B0 shows a dip near the prostate/rectum edge
opt.b0fun = @(r) 50*(1 -( 1/(1 + exp(-(r-10)/4)) + 1/(1 + exp((r-30)/4))));
% opt.b0fun = @(r) 3.5 ; % this is a constant half-pixel shift and leads to lots of
% Gibbs-type ringing
opt.Tfun  = @(itraj, k) (k-sz2DC(nhr))*1e-3  ;

% calc high res k-space yout
[yout, optout] = sysmatv(xinhr,'notransp', opt) ;

% calc low res k-space
optl.FOV = nlr ;
optl.verbose = false ; % Need these for LSQR
optl.scale = 'balanced' ;
optl.vec = true ;

optl.b0fun = opt.b0fun ;
optl.Tfun  = @(itraj, k) (k-sz2DC(nlr))*1e-3  ;

% calc low res k-space, this is from phantom(128) and not extracted from
% the high res k-space
[youtlr, optoutlr] = sysmatv(xinlr,'notransp', optl) ;


b0plot(optout) % plot the B0 field

% Extract central part of high res k-space
ind_cent = [sz2DC(nhr)-64:sz2DC(nhr)+63];
ncent = length(ind_cent) ;

disp(['Using k-space high res indices [ ',num2str(ind_cent(1)),' ... ', ...
    num2str(ind_cent(end)),' ]'])

ycent = yout(ind_cent) ;
cr1lr = ( [1:ncent] - sz2DC(ncent) ) * opt.FOV/ncent ; % image locations

figure('Name','k-space')
plot(optout.ck1, abs(yout), 'b', 'DisplayName', 'yout'), hold on
plot(optout.ck1(ind_cent), abs(ycent), 'r', 'DisplayName', 'ycent')
plot(optoutlr.ck1, abs(youtlr)*nhr/sqrt(nlr), 'DisplayName', 'youtlr')
legend


% Do LSQR on k-space from centre of high res
x0 = zeros(size(ycent)) ;

fmv = @(x,flag)sysmatv(x, flag, optl) ;

[x1, flag,relres,iter,resvec] = lsqr(fmv, ycent, [], [], [],[],x0) ;

% Now rpt but on kspace from a lower resolution image
[x1lr, flag,relres,iter,resvec] = lsqr(fmv, youtlr, [], [], [],[],x0) ;


figure('Name','prostate1d')
plot(optout.cr1, abs(k2i(yout)), 'DisplayName', 'k2i(yout): note ringing near shifts'), hold on
plot(cr1lr, abs(k2i(ycent)/nhr*nlr), 'DisplayName', 'k2i(ycent) using central k-space'), hold on
plot(cr1lr, abs(k2i(youtlr)*sqrt(nlr)), 'DisplayName', 'k2i(youtlr) using 128 point phantom')
plot(optout.cr1, xinhr,  'ro--', 'DisplayName', 'xinhr original'), grid on
%plot(cr1lr, xinlr,  'go--', 'DisplayName', 'xinlr')
plot(cr1lr, abs(x1)/nhr*sqrt(nlr), '--', 'DisplayName', 'CG recon of ycent')
plot(cr1lr, abs(x1lr), 'DisplayName', 'x1lr CG of lr')
xlabel('Position (mm)')
set(findobj(gcf,'Type','line'),'LineWidth',2)
title('Note ocillations in CG from ycent compared to lower res')
legend

% The CG recon has more, larger oscillations on the ycent data, compared to
% that based on the 128 points phantom. This was supposed to simulate using
% scanner data for recon (ycent) vs using an image to simluate distortion
% and then correcting.
% The simulation is not quite correct because of the way phantom(1024)
% differs from phantom(128), and, ycent is similar to a filtered version of
% the true k-space without taking into account alaising.
%
% Nevertheless, we can conclude that we can get large oscillations in a CG
% recon and attributing these to an incorrect B0 field is probably an
% unlikely explanation.
%


end


% ---
function b0plot(opt)
% B0PLOT Plot the B0 field used.
%
% b0plot(opt)
%
% opt.b0fun and opt.cr1 ust have been set
%

ncr1 = length(opt.cr1) ;

B0 = zeros([ncr1 1]) ;

for icr1 = 1: ncr1
    B0(icr1,1) = opt.b0fun(opt.cr1(icr1)) ;
end

figure('Name','B0')
plot(opt.cr1, B0)
xlabel('Position (mm)')
ylabel('B0 (Hz)')
end