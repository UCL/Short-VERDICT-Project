function N = normsb(Ds)
% normsb for the tensor field Ds
% i.e. if for example
% D is a (N x M x P) field of (d x d)
% tensors
% N is a (N x M x P) field of the 
% norms of the d x d tensor at corresponding position.
%
% From Phillip Batchelor
%


sizes = size(Ds);
l     = length(size(Ds));
df    = sizes(end);         % FIBRE DIMENSION
db    = l - 2;              % BASE DIMENSION

N = sqrt(sum(sum(Ds.*Ds,l),l-1));
