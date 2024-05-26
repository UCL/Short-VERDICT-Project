function B = rotn90(A,k)
%ROTN90 Rotate n-D matrix by 90 degrees
% Generalisation of ROT90
%
% B = rotn90(A,k)
%
% David.Atkinson@kcl.ac.uk   Guy's Hospital, London, UK.
%
% See also ROT90
%

m = size(A,1);
n = size(A,2) ;
nd = ndims(A) ;
highd = [3:nd ] ;

for idim = 1:nd
  indA{idim} = [1:size(A,idim)];
end

if nargin==1
  k=1;
else
  if length(k)~=1, error('k must be a scalar.'); end
  k = rem(k,4) ;
  if k<0
    k=k+4;
  end
end

if k==1
  A = permute(A,[2 1 highd(1,:) ]) ;
  indA{2} = indA{1} ;
  indA{1} = [n:-1:1] ;
  B = A(indA{:}) ;
elseif k==2
  indA{1} = [m:-1:1] ;
  indA{2} = [n:-1:1] ;
  B = A(indA{:}) ;
elseif k==3
  indA{1} = [m:-1:1] ;
  B = A(indA{:}) ;
  B = permute(B,[2 1 highd(1,:)]) ;
else
  B=A;
end


