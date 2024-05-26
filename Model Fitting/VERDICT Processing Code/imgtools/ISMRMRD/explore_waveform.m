function explore_waveform
% EXPLORE_WAVEFORM Waveform in ISMRMRD format
%
% David Atkinson  D.Atkinson@ucl.ac.uk
%
% See also MISST toolbox from UCL

% MISST toolbox creates a protocol that is [M 3K] where M is the number of
% measurements and K is the number of gradient points in each direction for
% one measurement. Within a row, stored x,y,z
%
% "The gradient waveform protocol.G is a M x 3K matrix which stores the
% values of the diffusion gradient at each time point. Each row represents 
% one measurement and contains the values of the gradients in x, y and z 
% direction. M is the total number of measurements and K is the number of 
% gradient points in each direction for one measurement:
% G1x(1) G1y(1) G1z(1) G1x(2) G1y(2) G1z(2) ... G1x(K) G1y(K) G1z(K)
% G2x(1) G2y(1) G2z(1) G2x(2) G2y(2) G2z(2) ... G2x(K) G2y(K) G2z(K)
% ...
% GMx(1) GMy(1) GMz(1) GMx(2) GMy(2) GMz(2) ... GMx(K) GMy(K) GMz(K)
% protocol.G includes the complete diffusion gradient and the user should 
% take into account any realistic situations such as the duration of the 
% rf pulses, crusher gradients, etc.
% protocol.G should satisfy the echo condition, that the integral of the 
% gradient over time should be 0. "

delta = 0.015; % duration in s
smalldel = delta - 0.005;  % duration in s
Nosc = 1;  % number of lobes in the oscillating gradient. A full period has 2 lobes
G = 0.08; % gradient strength in T/m;
tau = 1E-4; % time interval for waveform discretization
it = 0;
protocol_init.pulseseq = 'PGSE';
for i = 1:length(delta)
    for j = 1:length(Nosc)
        it = it+1;
        protocol_init.smalldel(it) = smalldel(i);
        protocol_init.delta(it) = delta(i);
        protocol_init.omega(it) = Nosc(j).*pi./smalldel(i);
        protocol_init.G(it) = G;
        protocol_init.grad_dirs(it,:) = [0.8 0.2 0]./sqrt(0.8*0.8 + 0.2*0.2); % gradient in x & y direction
    end
end
protocol_init.tau = tau;


protocolGEN.pulseseq = 'GEN';
protocolGEN.G = wave_form(protocol_init);
protocolGEN.tau = tau;
% include smalldel and delta as they make computation slightly faster
protocolGEN.delta = protocol_init.delta;
protocolGEN.smalldel = protocol_init.smalldel;

plot_G(protocolGEN)


function plot_G(protocolGEN)
% Plots gradients from MISST package

lw = 2 ; % LineWidth for plot
figure('Name','plot_G');
hold on
Gx = protocolGEN.G(1,1:3:end);
Gy = protocolGEN.G(1,2:3:end);
Gz = protocolGEN.G(1,3:3:end);

tau = protocolGEN.tau ;
times = [0:tau:(length(Gx)-1)*tau]*1E3 ;


plot( times ,Gx*1000,'LineWidth',lw)
plot( times, Gy*1000,'LineWidth',lw)
plot( times, Gz*1000,'LineWidth',lw)

xlabel('time (ms)','FontSize',16);
ylabel('Gx (mT/m)','FontSize',16);
set(gca,'FontSize',16);
grid on


