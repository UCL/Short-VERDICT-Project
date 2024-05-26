function [cbf, cbf_mask, p] = pcasl(SI_PD, SI_label, SI_ctrl, p )
% PCASL Pseudo continuous ASL calculation
% See Alsop et al MRM 2014
%
% [cbf, cbf_mask, params] = pcasl(SI_PD, SI_label, SI_ctrl, params)
%
%   SI_PD, SI_label, SI_ctrl  all need to be same size
%   params is a structure that can have any of the following fields:
%     lambda  (blood–brain partition coefficient mL/g)
%     T1_blood  ms
%     alpha  (labeling efficiency for PCASL)
%     label_dur ms
%     pld  (post label delay ms)
%
% Examples
% 1
%    cbf = pcasl_proc ;
%
% 2
%    dmain = datparse ; % Select main ASL file
%    [vasl,masl] = d2mat(dmain,{'slice','ilabtype','dyn'},'op','fp') ;
%    vasl = mean(vasl,5) ;
%     
%    dcalib = datparse ;
%    [vcalib,mcalib] = d2mat(dcalib,{'slice','dyn'},'op','fp') ;
%    vcalib = mean(vcalib,4) ;
%    cbf = pcasl(vcalib, vasl(:,:,:,1), vasl(:,:,:,2))
%
% D.Atkinson@ucl.ac.uk
% 

thresh = 15 ; % threshold for mask based on max PD image intensity

if ~isequal(size(SI_PD), size(SI_label), size(SI_ctrl))
    error(['Input images must be of the same size.'])
end

if nargin < 4
    p = [] ;
end
% Defaults from Alsop paper
if ~isfield(p,'lambda')
    p.lambda = 0.9 ;  % blood–brain partition coefficient mL/g
end
if ~isfield(p,'T1_blood')
    p.T1_blood = 1650 ; % 3T T1 (1350ms if 1.5T)
end
if ~isfield(p,'alpha')
    p.alpha = 0.85 ; % labeling efficiency for PCASL
end
if ~isfield(p,'label_dur')
    p.label_dur = 1650 ;
    disp(['Label Duration set to ',num2str(p.label_dur)])
end
if ~isfield(p,'pld')
    p.pld = 2000 ;
    disp(['Post Label Delay set to ',num2str(p.pld)])
end


% CBF in ml/100g/min
cbf = (6000 * p.lambda .* (SI_ctrl - SI_label) .* exp(p.pld/p.T1_blood) ) ./ ...
       ( 2 * p.alpha * p.T1_blood*1e-3 .* SI_PD .* (1-exp(-p.label_dur/p.T1_blood)) ) ;
   
cbf(~isfinite(cbf)) = 0 ;
loc = find(SI_PD < max(SI_PD(:))/thresh) ;

cbf_mask = ones(size(cbf)) ;
cbf_mask(loc) = 0 ;




   