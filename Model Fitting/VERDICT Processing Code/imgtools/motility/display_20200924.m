function display_20200924

% d2D = datparse(dselect('message','2D')) ;
d2D = dparse ;  % single frame
% Actually for 2D or 3D seqeunce

%d3D = datparse(dselect('message','3D')) ;


[v2D,m2D] = d2mat(d2D,{'slice','dyn'},'op','fp') ;
upl = 9000 ; % 2D
upl = 1900000 ; % 3D scan

%[v3D,m3D] = d2mat(d3D,{'slice','dyn'},'op','fp') ;

sls = 8 ;

fm2d = cat(1, ...
    cat(2,v2D(:,:,sls,:), v2D(:,:,sls+1,:), v2D(:,:,sls+2,:), v2D(:,:,sls+3,:) , v2D(:,:,sls+4,:)) , ...
    cat(2,v2D(:,:,sls+5,:), v2D(:,:,sls+6,:), v2D(:,:,sls+7,:), v2D(:,:,sls+8,:), v2D(:,:,sls+9,:)) , ...
    cat(2,v2D(:,:,sls+10,:), v2D(:,:,sls+11,:), v2D(:,:,sls+12,:), v2D(:,:,sls+13,:) , v2D(:,:,sls+14,:)) ) ;
     
  

[vfile, pvfile] = uiputfile('*.mp4','Select mp4 file', 'colon_motility.mp4') ;

vid = VideoWriter(fullfile(pvfile,vfile), 'MPEG-4') ;
vid.FrameRate = 7 ;

open(vid)
disp(['Writing to :',  fullfile(pvfile,vfile)])
writeVideo(vid, mat2gray(fm2d,[0 upl])) 
close(vid)
disp(['Written to :',  fullfile(pvfile,vfile)])


