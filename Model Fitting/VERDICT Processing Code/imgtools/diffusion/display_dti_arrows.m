function display_dti_arrows(EV,cmap,mask,BG_im)
% Display colour DTI tensors in 2D slice
% Jack Harmer, Nov 2012
%  -modified from RWC code
%
%  Usage:
%       display_dti_arrows(EV,cmap,mask,BG_im)
%
%   EV     = EigenValue Matrix
%   cmap  = Colour map for tensors
%   mask  = To mask tensors
%   BG_im = Background image

plot_white_tensors_v1 = 0;
plot_coloured_tensors = 1; 

arrow_length = 0.5;

X =  EV(:,:,1);
Y = -EV(:,:,2); %make sure diagonals are correct
Z =  EV(:,:,3);  % *** TODO : plot 3D tensors ***

% Plot background image
imagesc(squeeze(BG_im)); 
axis image; colormap gray;hold on;

%-----------------------------------------------------%
%          plot white tensors                         %
%-----------------------------------------------------%
if plot_white_tensors_v1
    q1 = quiver( X, Y,arrow_length,'Color',[1 1 1],'ShowArrowHead','off'); hold on;
    q2 = quiver(-X,-Y,arrow_length,'Color',[1 1 1],'ShowArrowHead','off'); hold on;
    set(q1,'LineWidth',2);
    set(q2,'LineWidth',2);
end

%-----------------------------------------------------%
%          plot coloured tensors                      %
%-----------------------------------------------------%
if plot_coloured_tensors        
    cmap(isnan(cmap))=0;
    quiver_colour(cmap,mask, X, Y,arrow_length); hold on;
    quiver_colour(cmap,mask,-X,-Y,arrow_length); hold on;
end

end
