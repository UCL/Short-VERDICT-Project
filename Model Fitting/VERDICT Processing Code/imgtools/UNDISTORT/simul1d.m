function simul1d
% SIMUL1D 1D simulation of signal in presence of B0 for different timings
%
%
% See also simul

% Units:
%  FOV  mm, sampled FOV
%  rpe  distance (r) in PE direction (mm)
%
%  
%  kpe k-space where 1 is the distance between lines, so 'true' value is
%  kpe/FOV in mm-1. Term in FT is exp(2.pi.i. k.y)
% 
%  B0 Hz,
%  tstep, time from one k-space line to next (s) (currently for fully
%  smpled)
%
%
%  
kpe = [-64:63] ;  % for kpe, a step of 1 corresponds to the sampled FOV 
nkpe = length(kpe) 

acq.nkpe = nkpe ;
% acq.b0pattern = 'flat1p' 'slope' 'g40'
acq.b0pattern = 'slope' ;
acq.tstep = 0.001 ;

acq.FOV = 200 ;

FOV = acq.FOV ;
rstep = 0.1;
rpe = [-FOV/2 : rstep: FOV/2-rstep] ;
nrpe = length(rpe) ;

ntraj = 1 ;
ncoil = 1 ;

disp(['FOV/nkpe = ',num2str(FOV/nkpe)])
disp(['A 1 pixel shift is ',num2str(1/nkpe/acq.tstep),' Hz.'])


figure
for irpe = 1: nrpe
    imm(irpe) = obj(rpe(irpe)) ;
    b0plot(irpe) =  B0_Hz(rpe(irpe), acq) ;
end
subplot(1,3,1), plot(rpe, imm), hold on
yyaxis right
plot(rpe, b0plot)



% i2k
S = zeros([nkpe, ntraj, ncoil]) ;
for ikpe = 1: nkpe
    for icoil = 1:ncoil
        for itraj= 1: ntraj
            for irpe = 1:nrpe
                y = rpe(irpe) ;
                S(ikpe, itraj, icoil) = S(ikpe, itraj, icoil) + ...
                    obj(y) * ...
                    exp(-2i*pi*(kpe(ikpe)*y/acq.FOV + (B0_Hz(y, acq) * T(itraj, kpe(ikpe), acq))) ) ;
            end
        end
    end
end
S = S * rstep ;

% Discrete FT k2i (not adjoint operation)
M = zeros([nrpe , ntraj, ncoil]) ;

for irpe = 1: nrpe 
    y = rpe(irpe) ;
    for icoil = 1:ncoil
        for itraj= 1: ntraj
            for ikpe = 1:nkpe
                M(irpe, itraj, icoil) = M(irpe, itraj, icoil) + ...
                    S(ikpe, itraj, icoil) * ...
                    exp(2i*pi*(kpe(ikpe)*y/acq.FOV)) ;
                    %exp(-2i*pi*(kpe(ikpe)*y/FOV + (B0_Hz(y) * T(itraj,kpe(ikpe)))) ) ;
            end
        end
    end
end
M = M /acq.FOV ;

% Adjoint operation. Thought process is 'how do I implement a matrix
% complex transpose using summation? Guides the conjugation and scaling.

% THis is "Conjugate phase ?"

Mcp = zeros([nrpe , ntraj, ncoil]) ;

for irpe = 1: nrpe 
    y = rpe(irpe) ;
    for icoil = 1:ncoil
        for itraj= 1: ntraj
            for ikpe = 1:nkpe
                Mcp(irpe, itraj, icoil) = Mcp(irpe, itraj, icoil) + ...
                    S(ikpe, itraj, icoil) * ...
                     exp(2i*pi*(kpe(ikpe)*y/acq.FOV + (B0_Hz(y, acq) * T(itraj,kpe(ikpe), acq))) ) ;
                 %   exp(2i*pi*(kpe(ikpe)*y/acq.FOV)) ;
            end
        end
    end
end
Mcp = Mcp /acq.FOV ;

% LSQR solution
% 
  
            

subplot(1,3,2), plot(kpe, abs(S(:,1,1)))
fftSlength = size(S,1) ;

vox_fftrecon = acq.FOV / fftSlength ;

rfft = -acq.FOV/2 + [0:(fftSlength-1)]*vox_fftrecon ;

% FT, viewed as a sum of narrow rectangles, should take the step size into account,
% In k2i, the step size is the k-space step (1/FOV) and the division by 1/N in the FFT 
% needs to be undone, leading to a correction that is N/FOV (=1/voxel
% size).
subplot(1,3,3), plot(rfft, abs(k2i(S)/vox_fftrecon), 'DisplayName','fftS') 
hold on, grid on
xlabel('mm')
plot(rpe, imm,'k', 'DisplayName','imm')

plot(rpe,abs(Mcp), 'DisplayName','Mcp')
legend

% Now simulate discretisation, then CG recon and stepped SENSE factor
% In discretisation exam, 'want' a B0 gradient that corresponds to more
% than a pixeo shift
%
% Take 



end % end function

% ----

% --------
function imval = obj(y)
% OBJ

if y<-50 
    imval = 0 ;
elseif y>=-50 && y< 50 
    % imval = 6*exp( -(y/20).^2 ) ;
    imval = 6 ;
else
    imval = 0 ;
end

end

% -------
function B0 = B0_Hz(y, acq)
switch acq.b0pattern
    case 'g40'
        
        B0 = 40*exp( -(y/20).^2 ) ;
    case 'slope'
        B0 = 10 / (1 + exp(-y/4)) ;
    case 'flat50'
        B0 = 50 ;
    case 'flat1p'
        B0 = 1/acq.nkpe/acq.tstep ;
        
    otherwise
        error(['Unkown B0 pattern'])
end
    
end

% -------
function t = T(itraj, ksp, acq)
    t = ksp * acq.tstep ;    
end
            