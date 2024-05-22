function [dist, signals] = RatioDistRician(A0, Ab, sigma0, opts)

% Function to construct ratio distribution of two Rician distributions
% (different means but same sigmas)


arguments

    A0 % b=0 mean signal
    Ab % b\=0 mean signal
    sigma0 % Noise standard deviation at TE=0

    % options

    opts.Nav_ratio = 1 % Ratio of signal averages between b\=0 and b=0

    % Range/Resolution of output pdf
    opts.zmin = 0
    opts.zmax = 2
    opts.dz = 0.01

    % Range/Resolution of integral
    opts.ymin = 0 
    opts.ymax = 2
    opts.dy = 0.01 % Made larger for speed


    % Calibration
    opts.calibrationfolder = "C:\Users\adam\OneDrive - University College London\UCL PhD\PhD Year 1\Projects\Short VERDICT Project\Code\Short-VERDICT-Project\Signal Simulation\Calibration Curves"


end


% == Construct array of zs
zs = linspace(opts.zmin, opts.zmax, ceil( (opts.zmax-opts.zmin)/opts.dz + 1) );

% == Construct array of ys (to integrate over at each z)
ys = linspace(opts.ymin, opts.ymax, ceil( (opts.ymax-opts.ymin)/opts.dy + 1) );

% == Construct first Rice distribution
RiceDist0 = makedist('Rician','s',A0,'sigma',sigma0);

% == Construct second Rice distribution

% Calibrate 
bsigma = (1/sqrt(opts.Nav_ratio))*sigma0;

% Calibrate mean and sigma after extra averaging (low SNR)
if and(Ab/sigma0<=5, opts.Nav_ratio>1)
    SNRs = load([char(opts.calibrationfolder) '/SNRs.mat'] ).SNRs;
    thisSNR = Ab/sigma0;
    thisSNRrounded = roundtowardvec(thisSNR, SNRs);
    load([char(opts.calibrationfolder) '/SNR ' num2str(thisSNRrounded) '/MeanCalibration.mat'])
    Ab = Ab*MeanDict(opts.Nav_ratio);

    load([char(opts.calibrationfolder) '/SNR ' num2str(thisSNRrounded) '/SigmaCalibration.mat'])
    bsigma = sigma0*MeanDict(opts.Nav_ratio);
end

% SNR 
% s = Ab/sigma0;
% if opts.Nav_ratio>1
%     Ab = Ab*sqrt(1+(1/s)^2);
%     bsigma = bsigma/sqrt(1+(1/s)^2);
% end

RiceDistb = makedist('Rician','s',Ab,'sigma',bsigma);

% Initialise array for ratio distribution
dist = zeros(length(zs),1);

% For each z value, integrate over y array
for zindx = 1:length(zs)

    z = zs(zindx);

    % Construct integrand
    integrand = (ys).*(RiceDistb.pdf(z*ys)).*(RiceDist0.pdf(ys));

    % Integrate integrad over ys
    integral = trapz(opts.dy,integrand);

    dist(zindx) = integral;


end

signals = zs;


end






function X=roundtowardvec(X,roundvec,type)
%function newnums=roundtowardvec(X,[roundvec],[type])
%
% This function rounds number(s) toward given values. If more than one
% number is given to round, it will return the matrix with each rounded
% value, otherwise it will return the single rounded value. It will ignore
% NaNs and return them back with NaNs.
%
% Inputs: X: the number(s) that you want rounded
%
%         roundvec:(opt) the values to round X to. If none given, it will
%           default to -inf:1:inf (and use the built in functions).
%
%         type:(opt) specifies which kind of rounding you want
%           the function to use.
%
%           Choices are: 'round' - round to nearest value
%                        'floor' - round toward -Inf
%                        'ceil'  - round toward Inf
%                        'fix'   - round toward 0
%                        'away'  - round away from 0 (ceil if positive and floor if negative)
%                     (see help files for more clarity)
%
%           If no type is given, the function will default to rounding to
%           the nearest value.
%
% Outputs: newnums: rounded values, in same shape as X input matrix
%          indices: indices of rounded values in roundvec
if nargin==0
	help roundtowardvec; %if nothing given, tell what to give
	return
elseif isempty(X)
	%if given empty, return empty without going through whole script
	return
end
if nargout>1
	error('Too many output variables are given');
end
if ~exist('type','var') || isempty(type)
	type='round';  %%round to nearest value if not specified
end
if ~exist('roundvec','var') || isempty(roundvec) || all(isnan(roundvec))
	if strcmpi(type,'round')
		%to nearest integer
		X=round(X);
	elseif strcmpi(type,'away')
		%nearest integer away from 0
		X=ceil(abs(X)).*sign(X);
	elseif strcmpi(type,'fix')
		%nearest integer toward 0
		X=fix(X);
	elseif strcmpi(type,'floor')
		%nearest integer toward -inf
		X=floor(X);
	elseif strcmpi(type,'ceil')
		%nearest integer toward inf
		X=ceil(X);
	else
		error('%sRound type not recognized. Options are:\n''round'' - round to nearest value\n''floor'' - round toward -Inf\n''ceil''  - round toward Inf\n''fix''   - round toward 0\n''away''  - round away from 0','')
	end
else
	%Ignore nan in roundvec
	roundvec(isnan(roundvec))=[];
	
	%Record which values are nan to ignore
	Xnan=isnan(X);
	
	%Hold onto size for returning value
	sz=size(X);
	
	%Calculate differences
	X=X(:);
	roundvec=roundvec(:)';
	diffs=bsxfun(@minus,X,roundvec);
	
	if strcmpi(type,'round') %to nearest value
		[~,inds]=min(abs(diffs),[],2);
		X=roundvec(inds);
	elseif strcmpi(type,'fix') %to nearest value toward 0
		
		iless=X<0;
		X(iless)=roundtowardvec(X(iless),roundvec,'ceil');
		X(~iless)=roundtowardvec(X(~iless),roundvec,'floor');
	elseif strcmpi(type,'ceil') %nearest value toward inf
		diffs(diffs>0)=nan;
		[~,inds]=min(abs(diffs),[],2);
		
		i_inf=X>max(roundvec);
		X=roundvec(inds);
		X(i_inf)=inf;
	elseif strcmpi(type,'floor') %nearest value toward -inf
		diffs(diffs<0)=nan;
		[~,inds]=min(abs(diffs),[],2);
		
		i_inf=X<min(roundvec);
		X=roundvec(inds);
		X(i_inf)=-inf;
	elseif strcmpi(type,'away') %nearest value away from 0
		
		iless=X<0;
		X(~iless)=roundtowardvec(X(~iless),roundvec,'ceil');
		X(iless)=roundtowardvec(X(iless),roundvec,'floor');
	else
		error('%sRound type not recognized. Options are:\n''round'' - round to nearest value\n''floor'' - round toward -Inf\n''ceil''  - round toward Inf\n''fix''   - round toward 0\n''away''  - round away from 0','')
	end
	
	%Return to output side
	X=reshape(X(:),sz);
	
	%Ignore nan in input dataset
	X(Xnan)=nan;
end
end