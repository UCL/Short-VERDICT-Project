% MATLAB script to display 

%% DRAG IMAGE INTO WORKSPACE

% Do that^


%% Define slice and range

slice = 5;

% ys = 60:110;
% xs = 55:110;

img = fIC(:,:,slice);


figure;
imshow(img, [0 1])
cb = colorbar;
a = cb.Position;
% set(cb,'Position',[a(1) a(2 a(3) a(4)])