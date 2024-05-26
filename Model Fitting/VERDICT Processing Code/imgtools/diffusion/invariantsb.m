function [v,Dm,D0] = invariantsb(Ds,iname)
% INVARIANTSB Vectorised computation of DWI FA, RA or mean
% v = invariantsb(Ds,iname)
% vectorised version of invariant
% computations. Exists only for 'simple'
% invariants, e.g.
% trace, norm, FA, RA
% which can computed in few '+, .* .^2'
% operations.
%
% iname - 'fa', 'ra','mean'
% EXAMPLE
%  
% philipp.batchelor@kcl.ac.uk
% $Id: invariantsb.m 143 2007-05-17 21:46:16Z ucacdat $
%

sizes = size(Ds);
l     = length(size(Ds));
df    = sizes(end);         % FIBRE DIMENSION
db    = l - 2;              % BASE DIMENSION


dind.type = '()';
for k = 1:l-2
  dind.subs{k} = ':';       % BASE COMPONENTS
end

%-------------------------------------------------------
% trace
%-------------------------------------------------------
dind.subs{l-1} = 1;         % FIBRE COMPONENTS
dind.subs{l}   = 1; % (1,1)-component

Dm = subsref(Ds,dind);     
for m = 2:df
  dind.subs{l-1} = m;
  dind.subs{l}   = m;
  Dm = Dm + subsref(Ds,dind);
end 
Dm = Dm/df;

%-------------------------------------------------------
% deviatoric
%-------------------------------------------------------

D0 = Ds;
for m = 1:df
  dind.subs{l-1} = m; 
  dind.subs{l}   = m;
  D0=subsasgn(D0,dind,subsref(Ds,dind) - Dm);
end 

%-------------------------------------------------------
% related invariants
%-------------------------------------------------------

switch iname,
 case 'fa',
  ws = warning('query','MATLAB:divideByZero') ;
  warning('off','MATLAB:divideByZero') ;
  v   = sqrt(df/(df-1))*(normsb(D0)./normsb(Ds));
  warning(ws.state,ws.identifier) ;
 case 'ra',
  v   = normsb(D0)./Dm;
 case 'mean',
  v   = Dm;
end
