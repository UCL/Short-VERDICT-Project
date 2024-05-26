function epglwi(pT2, T2s, TEs, tname)
% EPGLWI Modelling signal using EPG for LWI
%
%
% Uses EPG_T2 and not Shaihan's code

nTE = length(TEs);

T1 = 1500 ;
FA = [110 130 140 repmat(180,[1 nTE-3])] ;
%FA = [180 180 180 repmat(180,[1 nTE-3])] ;
%FA = [120 120 120 repmat(180,[1 nTE-3])] ;
%FA = [180 180 180 100 130 140 repmat(180,[1 nTE-6])] ; % Initial 180's allow for anatomic?


FA = FA/360*2*pi ;

[nT2, nx] = size(pT2) ;


dTE = unique(diff(TEs)) ;
if length(dTE) ~= 1
    error(['dTEs all have to be the same'])
end

% Try Shaihan's
% theta = [pi/9 FA(:).'] ;
% T2 = 60 ;
% [F0,Fn,Zn,F] = EPG_TSE(theta,dTE,T1,T2)



figure
for ix = 1:nx
    S = zeros(nTE,1) ;
    for iT2 = 1:nT2
      S = S + pT2(iT2,ix) * EPG_T2(nTE, dTE, T1, T2s(iT2), FA) ;
    end
    plot(TEs,abs(S),'DisplayName',['Tissue ',tname{ix}]), hold on 
end
grid on
axis([0 max(TEs) 0 1])
legend
