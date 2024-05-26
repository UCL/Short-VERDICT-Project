function [imout, refchop, vin_chop, rect] = mirt_call(refim, vin, rect)
%  MIRT_CALL Calls the MIRT Toolbox for 2D registration of multi-frame data.
%   Intended for cardiac diffusion data with ROI cropping
%  Handles scaling prior to call
%  [imout, refchop, vin_chop, rect] = mirt_call(refim, vin, rect)
%  [imout, refchop, vin_chop, rect] = mirt_call(refim, vin)
%
% Changed the number of levels from 3 to 2 as the cropped area for the LV is 
% quite small and the toolbox can sometimes fail on the levels with small
% sizes (NaNs from edge propagate in to leave no image).
% Can still suffer from failure (due to large trial displacements?)
%
% Uses the more recent MIRT Toolbox (7 May 2014), currently not the one in
% matlabgit/extern
%
% Example usage
% 
% [imout, refchop, vin_chop, rect] = mirt_call(refim, vin)
% eshow([vin_chop imout])
%
% David Atkinson  D.Atkinson@ucl.ac.uk
%

which mirt2D_register

if nargin > 2
    xx = rect.xx ;
    yy = rect.yy ;
else
    figure('Name','Select ROI')
    imshow(refim,[])
    h = imrect ;
    wait(h)
    pos = getPosition(h)  ;
    xx = round([pos(1):pos(1)+pos(3)]);
    yy = round([pos(2):pos(2)+pos(4)]);
    rect.yy = yy ;
    rect.xx = xx ;
end
disp(['ROI has width: ',num2str(length(xx)),' height: ',num2str(length(yy))])

refchop = refim(yy,xx) ;
vin_chop = vin(yy,xx,:);

[ny,nx,nf] = size(vin_chop) ;

imout = zeros([ny nx nf]) ;

main.similarity='rc';   % similarity measure, e.g. SSD, CC, SAD, RC, CD2, MS, MI  
main.alpha=0.05;        % similarity measure parameter (e.g., alpha of RC)
main.subdivide=2;       % use 3 hierarchical levels
main.okno=8;            % mesh window size, the smaller it is the more complex deformations are possible
main.lambda = 0.005;    % transformation regularization weight, 0 for none
main.single=1;          % show mesh transformation at every iteration

% Optimization settings
optim.maxsteps = 200;   % maximum number of iterations at each hierarchical level
optim.fundif = 1e-5;    % tolerance (stopping criterion)
optim.gamma = 1;       % initial optimization step size 
optim.anneal=0.8;       % annealing rate on the optimization step    

for iframe = 1:nf
    amin = min(min(vin_chop(:,:,iframe))) ;
    amax = max(max(vin_chop(:,:,iframe))) ;
    
    [res, newim] = mirt2D_register(mat2gray(refchop), ...
        mat2gray(vin_chop(:,:,iframe)), main, optim) ;
    imout(:,:,iframe) = newim*(amax-amin) + amin ;
end