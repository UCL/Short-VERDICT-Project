function lwi_non180
%lwi_non180
%   Detailed explanation goes here

% FA = [ 90 150 130 150 160 130 180 180 140 ]; dTE = 14 ;
FA = [   90 110 110 110 110 110 110 110 110 ]; dTE = 20 ;

TR = 1400 ;


LWFs = [0.05 0.1 0.15 0.2 0.4] ;
colors_lwf = { [0 0.4470 0.7410], [0.8500 0.3250 0.0980], [0.4660 0.6740 0.1880] , ...
    [0.3010 0.7450 0.9330], [0.9290 0.6940 0.1250]};

T2Ls = [150 200 400] ;
colors_t2l = { [0 0.4470 0.7410], [0.8500 0.3250 0.0980], [0.4660 0.6740 0.1880]} ;

%T2Ls = 200  ;
    
colors = { [0 0.4470 0.7410], [0.8500 0.3250 0.0980], [0.4660 0.6740 0.1880]};
T2Ss = [30 45 60 75] ;
markers_t2s = {'o','*','v','d'} ;

% B1s = [80 100 120] ;
B1s = 100 ;
markers = {'o','*','v'} ;

nrun = 1 ;
SNR = 60 ;

nLWF = length(LWFs) ;
nT2L = length(T2Ls) ;
nT2S = length(T2Ss) ;
nB1 = length(B1s) ;

hf = figure('Name','lwi_non180') ;
hf2 = figure;

for iLWF = 1: nLWF
    for iT2L = 1: nT2L
        for iT2S = 1 : nT2S
            for iB1 = 1: nB1
                for irun = 1:nrun
                    [slp, ba, endsig, firstsig] = calc_tse(LWFs(iLWF), B1s(iB1), FA, TR, dTE, T2Ss(iT2S), T2Ls(iT2L) ) ;
                    
%                     figure(hf)
%                     parg = {'Color',colors{iT2L},'Marker',markers{iB1}, 'MarkerFaceColor',colors{iT2L} } ;
%                     subplot(3,nLWF,iLWF), plot(T2Ss(iT2S), slp, parg{:}), hold on, axis([30 80 -9e-3 -3e-3])
%                     title(num2str(LWFs(iLWF))), grid on
%                     
%                     subplot(3,nLWF,iLWF+nLWF), plot(T2Ss(iT2S), ba, parg{:}), hold on, axis([30 80 0 14])
%                     grid on
%                     
%                     subplot(3,nLWF,iLWF+(2*nLWF)), plot(T2Ss(iT2S), ba/(abs(slp)), parg{:}), hold on
%                     grid on,  axis([30 80 0 2000])
                    
                    figure(hf2)
                    parg2 = {'Color',colors_t2l{iT2L}, 'Marker', markers_t2s{iT2S}, 'MarkerFaceColor',colors_t2l{iT2L} } ;
                    plot(LWFs(iLWF), firstsig/endsig, parg2{:} ), hold on, grid on
                    axis([0 0.5 0 12])
                end
            end
        end
    end
end

end

function [slp, ba, endsig, firstsig] = calc_tse(LWF, B1, FA, T1, deltaTE, T2S, T2L ) 

fa  = FA/360*2*pi ;
 
fa = fa * B1/100 ;

[F0L] = EPG_TSE(fa,deltaTE,T1,T2L) ;
[F0S] = EPG_TSE(fa,deltaTE,T1,T2S) ;

sig = LWF*abs(F0L)+ (1-LWF)*abs(F0S) ;

[p] = polyfit([ deltaTE*2 deltaTE*3 deltaTE*4 ], sig(2:4), 1) ;
slp = p(1) ;
endsig = (sig(7) + sig(8)) / 2 ;
firstsig = sig(1) ;

ba = (p(2)-endsig) / endsig ;

end



