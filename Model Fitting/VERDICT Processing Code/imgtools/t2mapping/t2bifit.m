function dfit = t2bifit(data, techo, varargin)
% SEE T2FITF!!!!
% T2BIFIT Fit T2 and luminal water fraction to data
%   Aim is to provide solid reference code 
%
% dfit = t2fit(data, techo, Name, Value, ...)
% dfit = t2fit(data, techo, props)
%
% data multi-echo data, last dimension must be necho
%      data converted to modulus double on input.
%
% techo  Echo Time. Either a scalar echo time difference, or, the 
%        each of the necho values (s)
%
% David Atkinson,  University College London
% D.Atkinson@ucl.ac.uk
%
%
% See also t2fit

% t2bound =  200e-3 ; % T2 boundary (s) between short and long T2s.

minT2    =   40e-3 ; % lower limit for short T2
shortT2max = 60e-3 ;

longT2min =  250e-3 ;
maxT2   =    350e-3 ; % upper limit for long T2


stT2short  =  50e-3 ; % starting short T2 
stT2long   = 300e-3 ; % starting long T2
stlwf  = 0.52   ; % starting luminal water fraction

data = double(abs(data)) ; % convert to modulus double in case complex or single
szdata = size(data) ;
szmap = szdata(1:end-1) ; % Maps are same size as image component of data.

necho = szdata(end) ;        % number of echoes
np = prod(szdata(1:end-1)) ; % number of pixels in data
rdata = reshape(data, [np necho]) ; % reshape to make looping and parfor easier

% Default mask
mask = ones([np 1]) ;

loc_mask = find(mask) ;
nloc = length(loc_mask) ;

mdata = rdata(loc_mask,:) ;  % just the pixels in the mask (helps parfor)

if length(techo) > 1
    TEs = techo ;
else
    TEs = [1:necho].*techo ;
end

if TEs(1) > 1
    warning(['Expecting TEs in s, dividing by 1000'])
    TEs = TEs ./ 1000 ;
end

% Check TEs are equally spaced (allow for small variations from DICOM.)
echo_thresh = 0.1e-3; % threshold for echo time separation difference (0.1 ms)
diffTEs = diff(TEs) ;
loc_uneven = find(abs(diffTEs - diffTEs(1)) > echo_thresh ) ;
if ~isempty(loc_uneven)
    warning(['Echo time separations differ by more than ',num2str(echo_thresh)])
end

% Preparation for parfor
% Note needed to place only masked pixels in mdata to help parallel aspect

fbi = cell(nloc, 1);  

disp(['Starting fit for ',num2str(nloc),' pixels.'])
% parfor lops over only the pixels in the mask.

% Fixed T2 fit.
sigs = mdata' ;
A = [exp(-TEs(:)/stT2short)  exp(-TEs(:)/stT2long) ] ;

ac = A\sigs ;
for iloc = 1:nloc
    fbi{iloc}.a = ac(1,iloc) ;
    fbi{iloc}.b = -1/stT2short ;
    fbi{iloc}.c = ac(2,iloc) ;
    fbi{iloc}.d = -1/stT2long ;
end


% % pw = PoolWaitbar(nloc, 'Fitting progress');
% % parfor iloc = 1: nloc
% %     increment(pw) % waitbar
% %     
% %     pdata = mdata(iloc,:) ;
% %     
% %     a = (1 - stlwf) * pdata(1) / exp(-TEs(1)/stT2short) ;
% %     c =       stlwf * pdata(1) / exp(-TEs(1)/stT2long)  ;
% %     
% %     % Note in exp2, it is -1/T2, hence ordering in upper and lower
% %     fopt = fitoptions('exp2', ...
% %         'Lower',        [ 0   -1/minT2       0  -1/longT2min    ] , ...
% %         'Upper',        [ inf -1/shortT2max   inf  -1/maxT2  ], ...
% %         'StartPoint',   [ a   -1/stT2short   c  -1/stT2long ] ) ;
% %     
% %     fbi{iloc}   = fit(TEs', pdata', 'exp2', fopt) ;  
% % end

% Collect outputs
dfit.szmap = szmap ;
dfit.nloc = nloc ;
dfit.loc = loc_mask ;

dfit.fbi = fbi ;




