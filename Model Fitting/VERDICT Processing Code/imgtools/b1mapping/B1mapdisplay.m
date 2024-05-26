function B1mapdisplay(B1)
% B1MAPDISPLAY Display B1 map ! Superceeded by B1DISPLAY
% See also B1DISPLAY

prompt={'Enter minimum B1 (%):', 'Enter max B1 (%):', 'Figure Name:'};
name='Input B1 display';
numlines=1;
defaultanswer={'50','150',inputname(1)};
answer = inputdlg(prompt,name,numlines,defaultanswer) ;
lb = str2num(answer{1}) ;
ub = str2num(answer{2}) ;
figname = answer{3} ;

B1(isnan(B1)) = 0 ;

figure('Name',figname)
imshow(B1,[lb ub])
colormap jet
colorbar