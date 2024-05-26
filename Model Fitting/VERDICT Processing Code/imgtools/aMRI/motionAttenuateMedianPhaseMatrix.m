function outMat = motionAttenuateMedianPhaseMatrix(inMat, varargin)
% motionAttenuateMedianPhaseMatrix  Matrix version (not video) adapted from
% video_phase_processing
%
%


%--------------------------------------------------------------------------
% Parameters

p = inputParser;

% Temporal window size
addOptional(p, 'WinSize', 11);

parse(p,  varargin{:});
win = p.Results.WinSize;

%--------------------------------------------------------------------------
vin = inMat ;
frame1 = vin(:,:,1) ;
[height, width] = size(frame1);
nFrames = size(vin,3) ;

% Video Code
% vr = VideoReader(inFile);
% frame1 = vr.read(1);
% [height, width, nChannels] = size(frame1);
% nFrames = vr.NumberOfFrames;
% vid = vr.read();

%--------------------------------------------------------------------------
% Pyramid functions
[h,w,nF] = size(vin);
[buildPyr, reconPyr] = octave4PyrFunctions(h,w);



nLevels = maxSCFpyrHt(frame1(:,:,1));
nOrients = 4;
filt = nOrients-1;
twidth = 1;

[~, pind] = buildPyr(frame1(:,:,1));
nElements = dot(pind(:,1),pind(:,2));

outMat = zeros(h,w,nF) ;

for ifr=1:nFrames
    fprintf('Processing frame %d of %d\n', ifr, nFrames);
    
    t0 = max(ifr-fix(win/2), 1); t1 = min(t0+win-1, nFrames);
    ts = t0:t1;
      
    % Read frames and transform
    frames = zeros(height, width, length(ts));
    pyrs = zeros(nElements, length(ts));
    for j = 1:length(ts)
        frame = vin(:,:,ts(j));
        
        % include here any scaling, typecasts
        
        frames(:,:,j) = frame;
        
        % Transform
        pyrs(:,j) = buildPyr(frame(:,:,1));
    end
   
    % Phase model
    meanPhase = median(angle(pyrs), 2);
    
    % Set all phases to the model and reconstrct
    pyr = pyrs(:,ts == ifr);
    amp = abs(pyr);
    phase = angle(pyr);
    for k = 1:size(pind,1)
        idx = pyrBandIndices(pind, k);
        phase(idx) = meanPhase(idx);
    end        
        
    % Reconstruct
    outMat(:,:,ifr) = reconPyr(exp(1i*phase) .* amp, pind);
        

end


