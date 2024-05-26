function D = interptensor(Ds, p)
% D = interptensor(Ds, p)
% Interpolate the tensor field Ds at the locations
% in p.
% 
% INPUT:
% Ds: a n1 x n2 x n3 x 3 x 3 tensor field.
% p : an array of N x 3 coordinates.
% 
% Donald Tournier re-implemented because
%  MatLab's interp3 is really slow.
% $Id: interptensor.m 144 2007-05-17 21:50:41Z ucacdat $


xp = floor(p); 
xp1 = xp + 1;
x = 1 - p(1) + xp(1);
y = 1 - p(2) + xp(2);
z = 1 - p(3) + xp(3);
D1 = Ds(xp(1),xp(2),xp(3),:,:);
D2 = Ds(xp1(1),xp(2),xp(3),:,:);
D3 = Ds(xp(1),xp1(2),xp(3),:,:);
D4 = Ds(xp1(1),xp1(2),xp(3),:,:);
D5 = Ds(xp(1),xp(2),xp1(3),:,:);
D6 = Ds(xp1(1),xp(2),xp1(3),:,:);
D7 = Ds(xp(1),xp1(2),xp1(3),:,:);
D8 = Ds(xp1(1),xp1(2),xp1(3),:,:);
    
D = reshape(D1(:)*x*y*z + D2(:)*(1-x)*y*z + D3(:)*x*(1-y)*z + ...
    D4(:)*(1-x)*(1-y)*z + D5(:)*x*y*(1-z) + ...
    D6(:)*(1-x)*y*(1-z) + D7(:)*x*(1-y)*(1-z) + D8(:)*(1-x)*(1-y)*(1-z),3,3);
