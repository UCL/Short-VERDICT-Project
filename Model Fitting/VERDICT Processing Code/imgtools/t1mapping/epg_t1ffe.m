function varargout = epg_t1ffe(varargin)
% EPG_T1FFE Extended phase graph spoilt gradient echo simulkation
% Adapted from Brian Hargreaves's epg_spgr.m
% See http://web.stanford.edu/~bah/software/epg/
%
%
% epg_t1ffe('param', value, ...)
%
% Parameters and defaults:
%  'FA' 15, 'TR' 10ms, 'T1' 1000ms, 'T2' 200ms
%  'flipphaseincrincr'|'incr' 117, 'Ntr' 200, 'Nstates' 20
%  'verbose' true unless outputs requested
%
% See also 

% Test for other Hargreaves files on path
if ~exist('epg_rf','file')
    warning(['Requires epg_rf and other files to be on path.'])
    disp(['Alter path or download from http://web.stanford.edu/~bah/software/epg/'])
end

if nargout == 0
    verbose = true ; % default, can be over-ridden by param,val pair
else
    verbose = false ;
end

% -- excitation

flipphase = 0;			% Start with flip angle along x (arbitrary)
flipphaseincr = 0;		% Phase increment 0 to start.
flipphaseincrincr = 117 ;	% Phase increment-increment of 117 deg.

% Defaults
Ntr = 1000 ;	% Number of TRs
Nstates = 20;	% Number of states to simulate (easier to keep constant
		% if showing state evolution.

flipang = 15; % Degrees
TR = 0.01;	% s
T1 = 1;		% s
T2 = 0.2;	% s


for ip = 1:2:length(varargin)
    var = varargin{ip+1} ;
    switch varargin{ip}
        case {'Ntr','NTR','nTR'}
            Ntr = var ;
        case 'Nstates'
            Nstates = var ;
        case 'FA'
            flipang = var ;
        case 'TR'
            TR = var ;
        case 'T1'
            T1 = var ;
        case 'T2'
            T2 = var ;
        case {'flipphaseincrincr', 'incr'}
            flipphaseincrincr = var ;
        case 'verbose'
            verbose = var ;
        otherwise
            warning(['Unknown parameter: ',varargin{ip}])
    end
end


P = zeros(3,Nstates);	% State matrix
P(3,1)=1;		% Equilibrium magnetization.

Pstore = zeros(2*Nstates,Ntr);	% Matrix to store for image of evolution.
Zstore = zeros(Nstates,Ntr);	% Matrix to store for image of evolution.


for k = 1:Ntr

  flipphase = flipphase + pi/180*(flipphaseincr);	% Incr. phase
  flipphaseincr = flipphaseincr + flipphaseincrincr;	% Increment increment!

  if k>300 & k < 310 
      flipang_this = flipang* (k-300)/10 ;
  else
      flipang_this = flipang ;
  end
  P = epg_rf(P,pi/180*flipang_this,flipphase);		% RF pulse
  s(k) = P(1,1);      			% Signal is F0 state.
  sd(k) = s(k)*exp(-1i*flipphase);	% Phase-Demodulated signal.

  % -- Keep states for evolution diagram
  Pstore(Nstates:2*Nstates-1,k) = P(2,:).'; % Put in negative states
  Pstore(1:Nstates,k) = flipud(P(1,:).');  % Put in positive, overwrite center.
  Zstore(:,k) = P(3,:).';

  % -- Simulate relaxation and spoiler gradient
  % Orig:  
  % 
  % 
  % 
  P = epg_grelax(P,T1,T2,TR,1,0,1,1); 
%   Tg = 0.0007 ;
%   k = 4258 * 2 * pi * 150 * 0.0007 ;  % see epg_sediff, here 15mT/m gradient for 0.7ms
%   D = 1e-9; % ADC 1000 mm2/s
%   
%   P = epg_grelax(P,T1,T2,TR-4*Tg,1,0,1,1);   % spoiler gradient, relaxation.
%   P = epg_grelax(P,T1,T2,Tg,k,D,1,1); 
%   P = epg_grelax(P,T1,T2,Tg,-k,D,1,1); 
%   P = epg_grelax(P,T1,T2,Tg,k,D,1,1); 
%   P = epg_grelax(P,T1,T2,Tg,-k,D,1,1); 
  
end;

if verbose
    figure('Name','plotstate');
    plotstate = cat(1,Pstore,Zstore);
    % dispim(plotstate,0,0.2);
    imshow(abs(plotstate),'Border','loose','DisplayRange',[0 0.2])
    xlabel('Time'); ylabel('F(top) and Z(bottom) States');
    
    figure('Name','magphase');
    titlestr = ['TR ',num2str(TR*1000),', T1 ',num2str(T1*1000),', T2 ',num2str(T2*1000), ...
        ', flipang ', num2str(flipang),', incr ',num2str(flipphaseincrincr)] ;
    epg_magphase(sd,'Ernst',ernstfn(T1, flipang, TR), 'title', titlestr);
end

if nargout >0
    varargout{1} = sd ;
end

