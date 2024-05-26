function S = ernstfn(T1, FA_deg, TR)
% S = ernstfn(T1, FA_deg, TR)

FA = FA_deg/360*2*pi ;

E = exp(-TR./T1) ;
        
S = sin(FA).*((1-E)./(1-cos(FA).*E)) ;