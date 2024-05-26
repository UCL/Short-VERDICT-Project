function vid2animgif
% VID2ANIMGIF Video to animated GIF
% vid2animgif
%   Asks for video input file and output gif name.
%   Displays rectangle on last frame to select FOV, double-click to accept.
%
% If you get a write permission error part way through, re run.
%
% David Atkinson  D.Atkinson@ucl.ac.uk  12 Nov 2012.
%

% LoopCount can be 0 (once), 1 (twice), .. Inf but for Windows should be in [1 65535]
LoopCount = 65535 ; 

%DelayTime in range 0 to 655, frame delay in seconds
prompt = {'Frame delay (s):','Delay Frame 1 (s):','Delay Last Frame (s)'};
dlg_title = 'GIF frame delays';
num_lines = 1;
def = {'0.1','0.1','0.1'};  % defaults
answer = inputdlg(prompt,dlg_title,num_lines,def);

[DelayTime status1] = str2num(answer{1}); 
[DelayFrame1 status2] = str2num(answer{2}) ;
[DelayFrameLast status3] = str2num(answer{3}) ;

prefnm = 'vid2animgif' ;

if ispref(prefnm,'filenm')
    filenm = getpref(prefnm,'filenm');
else
    filenm = [] ;
end

[fn,pn,FilterIndex] = uigetfile({'*','Video file'},'Select video filename',filenm) ;

if fn == 0
    warning(['No video file specified'])
    return
end

filenm = fullfile(pn,fn) ;
[vpn,vfn,vext] = fileparts(filenm) ;
setpref(prefnm,'filenm',filenm) ;

vidObj = VideoReader(filenm) ;
lastFrame = read(vidObj, inf); % see VideoReader documentation 
numFrames = vidObj.NumberOfFrames;
vWidth = vidObj.Width ;
vHeight = vidObj.Height ;

hf = figure('name','Select region for movie');
himshow = imshow(lastFrame,[]) ;
ha = gca ;
himrect = imrect(ha) ;
apos = wait(himrect) ;
xx = [max(1,round(apos(1))) : min(vWidth, round(apos(1)+apos(3)))] ;
yy = [max(1,round(apos(2))) : min(vHeight, round(apos(2)+apos(4)))] ;

if ispref(prefnm,'dirout')
    dirout = getpref(prefnm,'dirout');
else
    dirout = [] ;
end

if exist(dirout,'dir')
   DefaultName = [fullfile(dirout,vfn) '.gif'] ;
else
   DefaultName = [vfn '.gif'] ; 
end

[oFileName,oPathName] = uiputfile('*.gif','Animated GIF name',DefaultName);
if oFileName == 0
    warning(['No animated gif file specified'])
    return
end
setpref(prefnm,'dirout',oPathName) ;
agfn = fullfile(oPathName,oFileName) ;


hw = waitbar(0,'Writing animated gif','name','Video Read/Write') ;
for k = 1 : numFrames
    if ~strcmpi(vidObj.VideoFormat,'RGB24')
        warning(['Tested only for RGB24 videos.'])
    end
    
    frame = read(vidObj, k); 
    [M cmap] = rgb2ind(frame,256) ;
    if k==1
        imwrite(M(yy,xx), cmap, agfn,'gif','LoopCount',LoopCount,'DelayTime',DelayFrame1) 
    elseif k<numFrames
        imwrite(M(yy,xx), cmap, agfn,'gif','WriteMode','append','DelayTime',DelayTime) 
    else
        imwrite(M(yy,xx), cmap, agfn,'gif','WriteMode','append','DelayTime',DelayFrameLast)  
    end
    
    waitbar(k/numFrames,hw) ;
end
close(hw)

disp(['Written animated gif: ',agfn])
