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


end


% Construct array of zs
zs = linspace(opts.zmin, opts.zmax, (opts.zmax-opts.zmin)/opts.dz + 1);

% Construct array of ys (to integrate over at each z)
ys = linspace(opts.ymin, opts.ymax, (opts.ymax-opts.ymin)/opts.dy + 1);

% Construct first Rice distribution
RiceDist0 = makedist('Rician','s',A0,'sigma',sigma0);

% Construct second Rice distribution
bsigma = (1/sqrt(opts.Nav_ratio))*sigma0;
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