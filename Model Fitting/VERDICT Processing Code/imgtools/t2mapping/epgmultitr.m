function epgmultitr
% EPGMULTITR EPG for multiple TRs

TR = 2000 ; % ms
nTR = 10 ;
% Echos
dTE = 25 ; % delta TE
TE1 = 25 ; % first TE (needs to be dTE for Prony fit in lwi_fit)
necho = 8 ; % number of echos including first

TEs = [0:(necho-1)]*dTE + TE1 ; % these are times of 180's - not signal echo times?
nrefoc = length(TEs);              % not used anyway ...

T1 = 1500 ;
T2 = 200 ;

FA = [90 110 130 140 repmat(180,[1 nrefoc-3])] ; % Here include excitation 
%FA = [180 180 180 repmat(180,[1 nTE-3])] ;
%FA = [120 120 120 repmat(180,[1 nTE-3])] ;
%FA = [180 180 180 100 130 140 repmat(180,[1 nTE-6])] ; % Initial 180's allow for anatomic?


FA = FA/360*2*pi ;



dTE = unique(diff(TEs)) ;
if length(dTE) ~= 1
    error(['dTEs all have to be the same'])
end

% Use Shaihan's code but 1 slice here 
% how to evolve Z0 between TRs
Tshot = dTE*(nrefoc+0.5); % not really a 'shot' here 
Tdelay = TR - Tshot;
            
Xi = exp(-Tdelay/T1) ;

% L = [[-R1f-kf kb];[kf -R1b-kb]];
% C = [R1f*(1-f) R1b*f]';
% Xi = expm(L*Tdelay);
% I=eye(2);
% Zoff = (Xi - I)*inv(L)*C;
            
sig = [];
ts = [] ;

for iTR = 1:nTR
    if iTR == 1
        [F0,Fn,Zn,F] = EPG_TSE(FA,dTE,T1,T2) ;
        sig = [sig abs(F0)] ;
        ts = dTE*([1:nrefoc]*0.5);
    else
        % update z0. Code from Shaihan's test4_multislice_TSE.m
        %%% First take Z0 at end of TSE shot
        z0 = Zn(1,end);
                
        %%% Now evolve it by amount due to recovery period
        z0 = Xi*z0 + (1-Xi);
                
        [F0,Fn,Zn,F] = EPG_TSE(FA,dTE,T1,T2,'zinit',z0) ;
        sig = [sig abs(F0)] ;
        ts = [ts (iTR-1)*TR + dTE*([1:nrefoc]*0.5)]; 
    end
end
 
figure
plot(ts,sig)
grid
axis([0 max(ts) 0 1])



