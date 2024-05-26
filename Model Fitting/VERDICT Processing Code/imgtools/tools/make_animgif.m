function make_animgif(imgframes, outFilename, opts)
%MAKE_ANIMGIF Make an animated gif
%   
% make_animgif(imgframes,outFilename)
% make_animgif(imgframes,outFilename, Name=value, ...)
%
% Name-value pairs
%  DelayTime {1/10}
%  arange [amin amax] {[0 max(imgframes(:))]}
%  overwrite {false} | true
%
% Example
% make_animgif(imgframes,outFilename, DelayTime=1/25, arange=[0 1000])
%
% imgframes should be [ny nx nframes]
%
% David Atkinson
%

arguments
    imgframes (:,:,:) double
    outFilename 
    opts.DelayTime (1,1) {mustBePositive} = 1/10
    opts.arange (1,2) {mustBeNonnegative} = [0 max(imgframes(:))]
    opts.overwrite (1,1) logical = false
end

if exist(outFilename,"file") && opts.overwrite == false
    disp('File exists, to overwrite specify:  overwrite=true ')
    return
end

nf = size(imgframes,3) ;

for iframe = 1:nf
    [A,map] = gray2ind(mat2gray(imgframes(:,:,iframe), opts.arange),256) ;
    if iframe == 1
        imwrite(A, map, outFilename, "gif","LoopCount",Inf,"DelayTime",opts.DelayTime) 
    else
        imwrite(A, map, outFilename, "gif","WriteMode","append","DelayTime",opts.DelayTime)
    end 
end

disp(['Finished writing file: ',outFilename])

end