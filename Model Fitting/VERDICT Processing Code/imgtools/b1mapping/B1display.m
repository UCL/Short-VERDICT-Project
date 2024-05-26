function B1display(B1,anat, sl)
% B1DISPLAY  Display B1 map
%
%  B1display(B1)  B1 must be 2D image. Displays colormap with bar
%  B1display(B1,anat, sl)  B1 and anat can be volumes of the same size,
%  displays blended B1 map with anatomical, and anatomical alone.
%
% Example
%   
%   db1 = datparse ;
%   dt2 = datparse ;
%   [vt2, mt2] = d2mat(dt2,{'slice'},'op','fp') ;
%   [vb1,mb1] = d2mat(db1,{'slice','itype'},'itype',7,'op','fp') ;
%   [vb1r, mb1r] = dreslice(vb1,mb1,mt2) ;
%   B1display(vb1r, vt2, 100)
%
% D.Atkinson@ucl.ac.uk
%


prompt={'Enter minimum B1 (%):', 'Enter max B1 (%):', 'Figure Name:'};
name='Input B1 display';
numlines=1;
defaultanswer={'50','150',inputname(1)};
answer = inputdlg(prompt,name,numlines,defaultanswer) ;
lb = str2num(answer{1}) ;
ub = str2num(answer{2}) ;
figname = answer{3} ;

if nargin ==1
    
    B1(isnan(B1)) = 0 ;
    
    figure('Name',figname)
    imshow(B1,[lb ub])
    colormap jet
    colorbar
else
    if nargin == 2
        sl = 1 ;
    end
    if size(B1,3) ~= size(anat,3)
        error(['B1 and anat must have same number of slices'])
    end
    bm = mat2gray(B1(:,:,sl),[lb ub]) ;
    [X, map] = gray2ind(bm);
    rgb = ind2rgb(X,jet(length(map))) ;
    figure('Name',[figname,': B1 overlay'])
    imshowpair(anat(:,:,sl),rgb,'blend')
    figure('Name',['Anatomical sl: ',num2str(sl)])
    imshow(anat(:,:,sl),[]), imcontrast
end