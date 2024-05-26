function S = C2S(C,rlx, T10, M0, FA_rad, TR)
% C2S Gadolinium concentration to signal for spoilt gradient echo sequence
% S = C2S(C,rlx, T10, M0, FA_rad, TR)
%
% C concentration  [nt]
% rlx  Relaxivity 
% M0, T10  M0 and T1 baseline. T1 in same units as TR (ms)
% FA_rad [1] or [1 nt] 
% TR     [1] or [1 nt]
%
% S [nt]

C = C(:)' ;
nt = length(C) ;

if length(FA_rad) == 1
    FA_rad = repmat(FA_rad,[1 nt]);
end

if length(TR) == 1
    TR = repmat(TR,[1 nt]) ;
end

if length(TR)~= nt || length(FA_rad)~= nt
    error(['TR and FA_rad should have length nt: ',num2str(nt)])
end

T10 = repmat(T10,[1 nt]) ; % [1 nt]

% T1 to concentration relation
R1 = 1./T10 + rlx/1000*C ; % R1= 1/T1,  [1 nt]

% Ernst Function
% E = exp(-TR/T1) ;
% SI = M0 .* sin(FAs_rad).*((1-E)./(1-cos(FAs_rad).*E));
E = exp(-TR.*R1) ; % [1 nt]

S = repmat(M0,[1 nt]) .* sin(FA_rad) .* ((1-E)./(1-cos(FA_rad).*E)) ;

end