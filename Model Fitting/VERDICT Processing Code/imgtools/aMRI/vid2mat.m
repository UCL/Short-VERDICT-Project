function [mat, vr] = vid2mat
% VID2MAT Video file to matrix

vidFile = pref_uigetfile('vid2mat','vidFile') ;
[pn,fn,ext] = fileparts(vidFile) ;

vr = VideoReader(vidFile);
    
FrameRate = vr.FrameRate;    
mat = vr.read();

[h, w, nC, nF] = size(mat);

disp([' Read ',[fn,ext],' with ',num2str(nF),' frames (',...
    num2str(h),' x ',num2str(w),') and ',...
    num2str(nC),' channels.'])

