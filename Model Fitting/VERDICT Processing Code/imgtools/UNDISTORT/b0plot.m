function b0plot(opt, xin)
% B0PLOT Plot the B0 field used and the expected differential displacement
% 
% b0plot(opt, xin)
%
% opt.b0fun and opt.cr1 must have been set
%

ncr1 = length(opt.cr1) ;  % image coordinates

B0       = zeros([ncr1 1]) ;
B0modify = zeros([ncr1 1]) ;

for icr1 = 1: ncr1
    B0(icr1,1) = opt.b0fun(icr1) ;
    B0modify(icr1,1) = opt.b0funmodify(icr1) ;
end

figure('Name','B0')
plot(opt.cr1, B0, 'DisplayName','fwd B0'), hold on
plot(opt.cr1, B0modify, 'DisplayName','B0 in recon')
yyaxis right
plot(opt.cr1, xin, 'DisplayName', 'xin')
yyaxis left
xlabel('Position (mm)')
ylabel('B0 (Hz)')
legend, grid on
set(findobj(gcf,'Type','line'),'LineWidth',2)

dB0 = diff(B0) ;
figure('Name','b0plot: displacement differences') 
% code below adapted from sysmatv.m
pix = opt.cr1(2)-opt.cr1(1) ;
for iR = 1:length(opt.Ract)
    % tstep = opt.Tfun(1,opt.ckstart(iR)+1)-opt.Tfun(1,opt.ckstart(iR)) ;
    tstep = opt.tstep(iR) ;
    b0off = 1/(tstep/opt.Ract(iR))/length(xin) ;
%     disp(['B0 offset corresponding to 1 pixel (', ...
%            num2str(pix),'mm) : ', ...
%            num2str(b0off),' Hz. ( ',num2str(b0off/pix),' Hz/mm).'])

%   diff has one fewer point than B0. Place at mid points.
    plot(pix/2+(opt.cr1(1:end-1)), dB0/b0off, 'DisplayName', ...
        ['R ',num2str(opt.Ract(iR)),' tstep ',num2str(tstep*1000),' ms'])
    hold on
end
ylabel('-1 means pixel fully on top of neighbour')
xlabel('Position (mm)')
title('dB0/b0off'), legend, grid on
end