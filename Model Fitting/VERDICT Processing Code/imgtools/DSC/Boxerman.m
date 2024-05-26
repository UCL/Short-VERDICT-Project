function [K1, K2, CBVcorr] = Boxerman(Cref, Cmain, plotTF)
% BOXERMAN 
% [K1, K2, CBVcorr] = Boxerman(Cref, Cmain)
% [K1, K2, CBVcorr] = Boxerman(Cref, Cmain, plotTF)
%
% CBVcorr is not corrected for number of dynamics or temporal spacing
% 
% plotTF Plot results true {false} 
%
% David Atkinson
MINPOINTS = 10 ; % Min points in temporal profile
MAXSEP = 10 ; % Maximum points separating start of ref and main.

if nargin < 3
    plotTF = false ;
end

Cref = double(Cref) ; Cmain = double(Cmain) ;
Cmain(~isfinite(Cmain)) = 0;

ndr = length(Cref) ;
ndm = length(Cmain);

if ndr~= ndm
    minlen = min(ndr,ndm) ;
    if minlen < MINPOINTS
        K1=0; K2=0; CBVcorr = 0 ;
        return
    else
        if abs(ndr-ndm) > MAXSEP
           K1=0; K2=0; CBVcorr = 0 ;
           return 
        end
        % disp(['Ref and main not equal lengths - adjusting by ',num2str(ndr-ndm)])
        Cref = Cref(1:minlen) ;
        Cmain = Cmain(1:minlen) ;
    end
end

nd = length(Cmain) ;

cum = cumsum(Cref) ;

x0 = [1 0] ; % Initial guess at K1 and K2

opts = optimoptions('lsqnonlin','Display','off') ;
x = lsqnonlin(@(x)cf_boxerman(x,Cref,Cmain,cum), x0, [], [], opts) ;

K1 = x(1) ; K2 = x(2) ;

CBVcorr = trapz(Cmain + K2*cum) ;

if plotTF
    figure('Name','Boxerman')
    plot(Cref), hold on
    plot(Cmain)
    Ccorr = Cmain + K2*cum ;
    plot(Ccorr)
    title(['K1: ',num2str(K1),'. K2: ',num2str(K2)])
end


%------------------------------------------------------
function cf = cf_boxerman(x, Cref, Cmain, cum) 
% Cost funstion for lsqnonlin with Boxerman
% K1 = x(1)
% K2 = x(2)

cf = Cmain - ( x(1)*Cref - x(2)*cum) ;