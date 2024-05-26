function dout = addnoise(din, varargin)
% ADDNOISE Adds complex noise 
%   dout = adnoise(din, Name, Value, ...)
%
%   din can be image domain (any size) or k-space in 1st dimension only.
%   Function can estimate signal if image input and signal not specified.
%
%   If din is k-space, noise will be scaled to account for expected FT
%   (whether k2i or balanced using lsqr reconstruction with sysmatv)
%
%   dout = addnoise(din, var_denom)   OLD METHOD
%
% 
% Name, Value pairs:
%  'snr'      {10}            SNR (true image signal / noise std)
%  'signal'   {estimated}     Image domain true signal level. Estimated from 
%                             data if not supplied
%  'indomain' {'image'} | 'kspace'  Domain of input data
%  'scale'    {'balanced'} | 'i2k'  Scaling assumed for FTs, only needed if
%                                   indomain is kspace.
%  'verbose'  {true} | false 
%
%
% Examples
%  MRI = load('mri') ;
%  din  = double(squeeze(MRI.D(:,:,1,12))) ;
%  dout = addnoise(din, 'snr', 10, 'signal', 70) ;
%
%  
% 
% OLD USE
% dout = addnoise(din, var_denom)
% var_denom = 30 adds a respectable amount of noise to k-space.
% dout = din + rand(0 mean, variance 1)*data_max/var_denom +i*...
%
% See also addimnoise  sysmatv addnoiseTest

if nargin == 2
    % OLD CODE
    dmax = max(max(max(max(abs(din))))) ;
    varn = dmax / var_denom ;
    scale = varn ;
    dout = din + randn(size(din))*scale + 1i*randn(size(din))*scale ;
    return
end

snr = 10 ;
indomain = 'image' ;
scale =    'balanced' ;
verbose = true ;

for ipv = 1:2:length(varargin)
    param = varargin{ipv};
    val   = varargin{ipv+1} ;
    
    switch param
        case {'SNR', 'snr'}
            snr = val ;
        case 'signal'
            signal = val ;
        case 'indomain'
            indomain = val ;
        case 'scale'
            scale = val ;
        case 'verbose'
            verbose = val ;
        otherwise
            error(['Unknown parameter: ',param])
    end
end

% 1) Determine standard deviation of noise to add.
% 2) Scale if noise added in k-space prior to Fourier Transform.

if ~exist('signal','var')
    % estimate signal
    switch indomain
        case 'image'
            top = prctile(abs(din(:)),99) ;
            jimg = abs(din) > top/20 ;
            signal = median(din(jimg)) ; % take signal as median value of image 
                                         % excluding low values (assumed air)
                                         
            if verbose
                disp(['Estimated true image signal level: ',num2str(signal)])
            end
            
        otherwise                        
            error(['Please provide image domain signal value'])
    end
end

switch indomain
    case 'image'
        scalef = 1;
    case {'kspace', 'k-space'}
        % set scaling factor for added noise so that it gives correct SNR
        % in the image domain
        
        switch scale
            case {'i2k','k2i'} % my i2k and k2i routines
                scalef = sqrt(size(din,1)) ;
                if size(din,2) > 32
                    warning('Assumed din was k-space with 2nd dimension coils')
                end
            case 'balanced'
                scalef = 1;
            otherwise
                error('FFT scaling method not implemented')
        end
    otherwise
        error('indomain must be image or kspace')
end

stdn = signal / snr ; % Noise standard deviation

szdin = size(din) ;

noise = stdn.*randn(szdin) + 1i*stdn.*randn(szdin) ;

dout = din + scalef.*noise ;

% if verbose 
%     disp(['Standard deviation of noise applied: ',num2str(stdn)])
%     if strcmp(indomain,'image')
%         pdRa = fitdist(abs(dout(:)),'Rician')
%     end
% end  


