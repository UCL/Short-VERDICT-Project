function hh = quiver_colour(my_colormap,mask,varargin)
% Coloured quivers...
% Jack Harmer, Nov 2012


%TODO make these args in
lw=2;
scale_factor=0.6; 

% Check input arguments
if (nargin-2) < 4
  [msg,x,y,u,v] = xyzchk(varargin{1:2});
else
  [msg,x,y,u,v] = xyzchk(varargin{1:4});
end

% Scalar expand u,v
if prod(size(u))==1
    u = u(ones(size(x))); 
end

if prod(size(v))==1
    v = v(ones(size(u))); 
end

% Increase length of lines
u=u*scale_factor;
v=v*scale_factor;

%----------------------------------------------
% Make velocity vectors and plot them
x = x(:).';
y = y(:).';
u = u(:).';
v = v(:).';

uu = [x;x+u;repmat(NaN,size(u))];
vv = [y;y+v;repmat(NaN,size(u))];

uui=uu(:);  
vvi=vv(:);  
imax=size(uui);

r=my_colormap(:,:,1);
g=my_colormap(:,:,2);
b=my_colormap(:,:,3);

hold on
c=0;
% tic
for i=  1:3:imax-1
    c=c+1;
    if mask(c)
        % Much faster than plot
        line(uui(i:i+1),vvi(i:i+1),'linewidth',lw,'color',[r(c) g(c) b(c)]);
    end
end
% toc

hh=gca;
set(gca, 'color', [0 0 0],'Xcolor','w','Ycolor','w');
set(gca,'XTick',[],'YTick',[])
set(gcf, 'color', [0 0 0]);

