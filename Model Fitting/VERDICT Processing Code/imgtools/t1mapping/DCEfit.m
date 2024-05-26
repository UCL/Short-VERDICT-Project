function fit = DCEfit(T10fit, Smeas,Stimes, FA_rad, TR)
% DCEfit Fit of DCE pharmacokinetic parameters.
%
% DCEfit(T10fit, Smeas, Stimes, FA_rad, TR)
%
% T10fit structure output from MFAT1 with fields
%   Smeas [N nt]
%   Stimes [1 nt]
%   FA_rad [1 1] or [N 1] or [N nt] the B1 corrected flip angles for the 
%      contrast sequence (not the T10 basleine mapping)
%   TR [1 1] or [1 nt] the TRs for the contrast sequence
% 
%
% Example
%   T10fit = MFAT1(MFA, FAs_deg, TRs, B1map) ;
%   DCEfit(T10fit, Smeas,Stimes, FA_rad, TR) ;
%
% David Atkinson  D.Atkinson@ucl.ac.uk
% See also C2S pharma MFAT1 DATPARSE D2MAT
%

options = optimset('Display','off') ; % lsqnonlin options.
rlx = 4.9   % s-1 mM-1 Relaxivity of contrast agent
model.AIFmethod = 'Parker' ; % Parker population AIF
model.DCEmethod = 'ETM' ; % Extended Tofts Model (with vp). 
% Unknowns x are Ktrans, ve, vp and optionally tonset
% if x is length 3, user must supply model.tonset

x0 = [ 0.5/60 0.6 0.3 10] ; model.tonset = 0 ;

% Check and adjust inputs
[N nt] = size(Smeas) ;
if size(FA_rad,1)==1
    FA_rad = repmat(FA_rad,[N 1]) ;
end
if size(FA_rad,2) == 1
    FA_rad = repmat(FA_rad,[1 nt]) ;
end
if size(TR,2)==1
    TR = repmat(TR,[1 nt]) ;
end
if size(Stimes,2) ~= nt
    error(['Number of Stimes should match input data'])
end
% T10fit can have fields: T1_lin, T1_nl, M0_lin, M0_nl, resn_lin, resn_nl
if isfield(T10fit,'T1_nl')
    T10 = T10fit.T1_nl ;
    M0 =  T10fit.M0_nl;
else
    T10 = T10fit.T1_lin ;
    M0  = T10fit.M0_lin ;
end

fit = zeros([N length(x0)]) ;
for ip = 1:N
    %loop over pixels in Smeas. 
    
    fhobjf = @(x)objf(x, Smeas(ip,:), Stimes, model, rlx, T10(ip), ...
        M0(ip), FA_rad(ip,:), TR) ;
    
    [x,resnorm,residual,exitflag] = lsqnonlin(fhobjf,x0,[],[],options) ;
    fit(ip,:) = x ;
end
end

    
function resid = objf(x,Smeas,Stimes, model, rlx, T10, M0, FA_rad, TR)
% OBJF Objective function for DCE pharmacokinetic fit
C = pharma(x, Stimes, model) ;
resid = Smeas - C2S(C, rlx, T10, M0, FA_rad, TR) ;
end
    
 
 
 
 