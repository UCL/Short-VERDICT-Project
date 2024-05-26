function [evec, eval] = d2eig(D)
% d2eig Single tensor to sorted eigenvectors and values
%
% [evec, eval] = d2eig(D)
%
% D      single tensor [3x3] but input will be squeezed to enable calling with
%        Dvol(i,j,k,:,:)
% evec   [3 x 3 ] eigenvectors in DESCENDING order (principal first)
%                 eigenvecs are in the columns e.g. ev1 = evec(:,1) ;
% Note in the vector it is 1st dim, 2nd dim, 3rd dim which corresponds in
% image space to yxz.
% eval   [1 x 3] eigenvalues
%
% David Atkinson D.Atkinson@ucl.ac.uk
% $Id: d2eig.m 144 2007-05-17 21:50:41Z ucacdat $

[ev,lm] = eig(squeeze(D)) ;
[eval, ord] = sort([lm(1,1), lm(2,2), lm(3,3)],2,'descend') ;
evec = [ev(:,ord(1)), ev(:,ord(2)), ev(:,ord(3)) ] ;
