function s = simulset(varargin)
% SIMULSET  Sets structure for joint_simul
%
% simulopt = simulset(param, val, ...)
%
% D.Atkinson@ucl.ac.uk
%
% See also JOINT_SIMUL


% defaults
s.name = 'default' ;
s.verbose = false ;

s.nfe = 32 ;
s.ntraj = 2 ; % up and down

s.npe_u = 17 ; % number of phase encodes accounts for SENSE but not partial F
s.npe_f = 31 ; % number of frequency encodes

s.partialF = 0.7 ; % approximate partial Fourier factor

s.addnoise = true ; 
s.snr = 20 ;

% Coils
s.csens_obj_only = true ; 
s.ncoil = 4 ; % should only be 1 (uniform sens) or 4
s.cph_mult = 10 ; % coil phase step in degrees

% B0 field
s.B0_amp = 100 ;  % amplitude of gaussian B0 pattern

s.time_seg = false ; % Boolean for time segmented recon
s.x_check_tseg = false ;  % checks time segmentation code against DFT

s.maxit = 5 ; % Passed to LSQR 
s.tol = [] ;
s.x0 = 'empty' ; % 'empty' for all zeros, or use 'img' for oracle answer

% Pre-conditioning thresholds
s.sens_lower_thresh = 1/10000 ;
s.pc_thresh = 1/10000 ;

% Joint Recon
s.B0_joint = false ;
s.x0_joint = true ;
s.check_linear = false ; % check linear approx valid 
s.B0_offset = 0 ; % constant offset in assumed B0
s.B0_scale = 1 ;  % scaling applied to true B0 as input to joint
s.nouter = 5 ;    % number of outer iterations for image + B0 
s.B0_project_real = false ; % project B0 to real
s.B0_opt = 'lsqr' ; % 'lsqr' or 'fminsearch'  (for global linear correction)

% Sequence time between phase encodes acquired
s.dt_s = 1e-3 ;
s.es = [0 0] ; % echo time shift for each traj

for ipv = 1:2:length(varargin)
    param = varargin{ipv};
    val = varargin{ipv+1} ;
    if ~isfield(s, param)
        error(['Param: ',param,' is not a field of s.'])
    end
            
    s.(param) = val ;
end



