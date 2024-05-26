function tr = ft(Ds,x0,u0,varargin)
% tr = ft(Ds,x0,u0,Rg2rcs,ds,nmax,famin)
% 
% Do fibre tracking on tensor field Ds
% starting at x0, with 'hint' direction u0.
% 
% INPUT:
%   Ds: a tensor field, size n1 x n2 x n3 x 3 x 3
%   x0: a position           1  x 3
%   u0: a direction vector   3  x 1 
% 
% OPTIONAL INPUT:
%   Rg2rcs: [3x3] rotation matrix from gradient coords to matrix rcs.
%           This is to relate gradient coord system to matrix.
%   ds: step size: 0.1
%   nmax: max number of iters: 1000)
%   famin: FA threshold: 0.2
% OUTPUT:
%   tr: coordinates of ft,   3 X n  , where  n <= nmax=1000
%
% Current algorithm is Donald Tournier's implementation
% of ?? (Conturo/Mori)
% 
% STOPPING CRITERIA:
% 
%   if fa <= famin
%   if reaches boundary
%   if nsteps reaches nmax 
% 
% REFERENCES:
%   Mori??
% 
% Adapted from Philipp Batchelor by David Atkinson
% D.Atkinson@ucl.ac.uk
% $Id: ft.m 215 2008-11-21 16:17:15Z ucacdat $
%=======================================================
% guess ds?

defaults = {eye(3),1/10,10000,0.15};
neIdx   = ~cellfun('isempty',varargin);
defaults(neIdx) = varargin(neIdx);
[Rg2rcs,ds,nmax,famin] = deal(defaults{:});
u0 = u0/norm(u0);
%-------------------------------------------------------
dim = [size(Ds,1) size(Ds,2) size(Ds,3)]';

D = interptensor(Ds,x0);
fa0 = invariantsb(D,'fa') ;
%-------------------------------------------------------
[U,S,V] = svd(D);
v = U(:,1);

v = Rg2rcs * v ;
v = v ./norm(v) ; % Rg2rcs can have scaling but stepsize should be determined by ds

if v'*u0 < 0
  v = -v;
end
tr = repmat(nan,3,nmax);
tr(:,1) = x0(:);
tr(:,2) = x0(:) + ds*v;
lastd = v;

%-------------------------------------------------------
j = 2;
x = tr(:,2);

while (x >= 1 & x <= dim & j < nmax ) % within data and less than max steps
  D = interptensor(Ds,x);
  if invariantsb(D,'fa') < famin
    disp(['FA is ',num2str(invariantsb(D,'fa'))]);break; 
  end
  [U,S,V] = svd(D); v = U(:,1);
  
  v = Rg2rcs * v ;
  v = v./norm(v) ;
  
  f = 2*((x - tr(:,j-1))'*v >= 0) -1 ;
  if dot(f*v,lastd) < 0.7071
      disp(['Terminate on angle ']); break
  else
      lastd = f*v ;
  end
  
  tr(:,j+1) = x + f*ds*v;   % adaptative stepsize
  j = j + 1;
  x = tr(:,j);
end
tr(:,j+1:nmax) = [];

return
