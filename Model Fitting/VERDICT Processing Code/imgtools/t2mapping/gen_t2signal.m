function [S, TEs, tname] = gen_t2signal
% GEN_T2SIGNAL Generate a signal from multi echo data
%
% [S, TEs] = gen_t2signal
%
% See also lwi_fit

% Echos
dTE = 25 ; % delta TE
TE1 = 25 ; % first TE (needs to be dTE for Prony fit in lwi_fit)
necho = 8 ; % number of echos including first

TEs = [0:(necho-1)]*dTE + TE1 ;

% Tissue T2 probability distribution on a discrete grid
np = 2000 ;
rangeT2 = [ 1 2000] ; % ms


T2s = linspace(rangeT2(1), rangeT2(2), np) ;

% Loop over examples
nexam_max = 10 ;

ix = 0 ; % running total
pT2 = zeros(np, nexam_max) ;

% Example 1

% Include tissue classes (could be spikes or Gaussian distributions)
% ix = ix + 1 ;
% pT2(40,ix) = 1 ; % index of 40 here coressponds to 40ms 
% pT2(300,ix) = 0.2 ;
% 
% % Example 2
% ix = ix + 1 ;
% pT2(40,ix) = 1;
% pT2(300,ix) = 0.1;
% 
% ix = ix + 1 ;
% pT2(45,ix) = 1;
% pT2(300,ix) = 0.1;
% 
% ix = ix + 1 ;
% pT2(40,ix) = 1;
% pT2(202,ix) = 0.1;

ix = ix + 1 ; tname{ix} = 'H1' ;
pT2 = addgaussian(pT2,T2s, ix,  40, 5, 5) ;
pT2 = addgaussian(pT2,T2s, ix, 300, 5, 1) ;

ix = ix + 1 ; tname{ix} = 'H2' ;
pT2 = addgaussian(pT2,T2s, ix,  40, 5, 1) ;
pT2 = addgaussian(pT2,T2s, ix, 300, 5, 1) ;

ix = ix + 1 ; tname{ix} = 'T' ;
pT2 = addgaussian(pT2,T2s, ix,  40, 5,10) ;
pT2 = addgaussian(pT2,T2s, ix, 300, 5, 0.1) ;

% ix = ix + 1 ;
% pT2(37,ix) = 0.7 ;
% pT2(131,ix) = 0.17;

% % Example 3
% ix = ix + 1 ;
% pT2(40,ix) = 1;
% pT2(1200,ix) = 0.2;
% 
% ix = ix + 1 ;
% pT2(40,ix) = 1 ; % index of 40 here coressoponds to 40ms 
% pT2(300,ix) = 0.1 ;
% 
% % Example 
% ix = ix + 1 ;
% pT2(40,ix) = 1 ;
% pT2(600,ix) = 0.1;
% 
% % Example 
% ix = ix + 1 ;
% pT2(40,ix) = 1;
% pT2(1200,ix) = 0.1;
% 
% ix = ix + 1 ;
% pT2(35,ix) = 1;
% pT2(1200,ix) = 0.1;

nexam = ix ;

% Ensure pT2s have a cumulative probability of 1
ptot = sum(pT2,1) ;
pT2 = pT2./repmat(ptot,[np 1]) ;

% Compute LWF
lwfT2 = 200 ; % LWF cut off T2 (ms) ;

[~,icutoff] = min(abs(T2s-lwfT2)) ;
figure
for iex = 1:nexam
    low = sum(pT2(1:icutoff,iex)) ;
    high = sum(pT2(icutoff+1:end, iex)) ;
    LWF(iex) = high / (low + high) ;
    tname{iex} = [tname{iex},' ',num2str(LWF(iex))] ;
    
    plot(T2s, pT2(:,iex)), hold on
end
xlabel('T2')

pT2 = pT2(:,1:nexam) ;

mp = max(pT2(:));
plot([T2s(icutoff) T2s(icutoff)],[0 mp],'--') % marker for LWI cutoff

epglwi(pT2, T2s, TEs, tname)

% Compute T2 signal
figure
for iex = 1:nexam
    S = zeros(1,necho) ;
    for iecho = 1: necho
        for iT2 = 1:np
            S(iecho) = S(iecho) + pT2(iT2,iex)*exp(-TEs(iecho)/T2s(iT2)) ;
        end
    end
    
    scale = 1/100 ;
    noise = randn(size(S))*scale + 1i*randn(size(S))*scale ;
    S = S + noise ;
    S = abs(S) ;
    
    plot(TEs,S, 'DisplayName',tname{iex}), hold on 
end

axis([0 max(TEs) 0 1]), grid on
xlabel('TE')
legend
disp(['LWFs: ',num2str(LWF)])



function pT2 = addgaussian(pT2,T2s, ix, u, sd,a)
for iT2 = 1:length(T2s)
    pT2(iT2,ix) = pT2(iT2,ix) + a*exp(-0.5*((T2s(iT2)-u) / sd).^2) ;
end



