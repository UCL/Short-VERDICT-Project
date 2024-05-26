function [outputArg1,outputArg2] = tse_learn(inputArg1,inputArg2)
%TSE_LEARN Learn about EPG in TSE
%   Detailed explanation goes here
outputArg1 = inputArg1;
outputArg2 = inputArg2;

% Set up tissues
tissue = struct('T1',{1400, 1400}, ...
    'T2',{60, 300}, ...
    'T2b', {12e-6, 12e-6},...
    
    ) ;


nrefocus = 25 ;

% 90 180  180
FA_rad = d2r([90 180*ones([1 nrefocus])]) ;
b1sqrdtau = [32.7 213.1*ones(1,nrefocus)]; % uT^2 ms  from Shaihan
df = 10.9 * 42.57e3 * 6e-3; % G * gamma * dx


seq = struct('label','TSE','ESP',7.7, 'nslice',1,'nrefocus', nrefocus, ...
    'nTR', 4,'TR',5000, 'FA_rad',FA_rad) ;






% Set up sequence parameters

% 
end

