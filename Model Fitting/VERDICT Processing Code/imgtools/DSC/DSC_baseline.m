function [mean_base, SD_base, bstart, bend] = DSC_baseline(pd, varargin)
% [mean_base, SD_base, bstart, bend] = DSC_baseline(pd, param, val,...)
%
% pd [npix ndyn]
%
% Parameters and defaults
%  bstart [2] Starting dynamic (previous ignored)
%  bend   [5] Last baseline dynamic (updated unless auto_bend is false)
%  auto_bend [true]
%  bstd_fac [3] Multiple of baseline std for detection of bolus arrival
%  plotb [true] Plot data
%
% David Atkinson
%

MIN_DYN = 30 ;

if size(pd,2) == 1
    pd = reshape(pd,[1 length(pd)]) ;
end

[npix, ndyn] = size(pd) ;


if ndyn < MIN_DYN
    warning(['Number of dynamics is less than ',num2str(MIN_DYN)])
end

pdm = squeeze(mean(pd,1)) ; % average pixel dynamics over ROI

% defaults
bstart = 2 ;
bend = 5 ;
auto_bend = true ;
bstd_fac = 3 ;
plotb = true ;

for ip = 1:2:length(varargin)
    switch varargin{ip}
        case 'bstart'
            bstart = varargin{ip+1} ;
        case 'bend'
            bend = varargin{ip+1} ;
        case 'auto_bend'
            auto_bend = varargin{ip+1} ;
        case 'bstd_fac'
            bstd_fac = varargin{ip+1} ;
        case 'plotb'
            plotb = varargin{ip+1} ;
        otherwise
            error(['Unknown input parameter: ',varargin{ip}])
    end
end

if bend > ndyn
    error(['Baseline end is greater than number of dynamics'])
end

if auto_bend
    bend_found = false ;
    for iend = bend:ndyn-1
        basel = pdm(bstart:iend) ;
        
        smean = mean(basel) ;
        sstd = std(basel) ;
        
        if pdm(iend+1)< smean - bstd_fac*sstd 
            bend = iend ;
            bend_found = true ;
            break  ;
        end
    end
    % if ~bend_found, warning('No baseline end found'), end
end
        
mean_base = mean(pdm(bstart:bend)) ;
SD_base = std(pdm(bstart:bend)) ;

if plotb
    figure('Name','Baseline')
    plot(pdm), hold on
    plot([bstart bend],[mean_base mean_base])
    plot([bstart bend],[mean_base-SD_base  mean_base-SD_base])
    plot([bstart bend],[mean_base-bstd_fac*SD_base  mean_base-bstd_fac*SD_base])
end
