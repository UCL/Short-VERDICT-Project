function varargout = geomchecker(varargin)
% Checks the geometry information
% Aimed at Philips EPI
%
%  geomchecker(param, value ,... )
%
% param, val:  
%   'msg', message     string for message 
%   'MRecon', MR
%   'labfn',  labfn    label filename (calls ugetfile if 
%                      file does not exist)
%   'sininfo', info    info struct form*
%   'geom', geom       geom struct
% 
% * not yet implemented. Intentiion is to check sin parameters == RC. In
% future also look to check .LIST parameters - to be independent of patch
% and also for future ISMRMRD conversion.
%
%
% David Atkinson  D.Atkinson@ucl.ac.uk
%
% See also recon_process ori_info 

inpMrecon = false ;
inpsininfo = false ;
inpgeom = false ;
msg = '' ;

for ipv = 1:2:length(varargin)
    val = varargin{ipv+1} ;
    param = varargin{ipv} ;
    
    switch param
        case {'MRecon', 'Mrecon'}
            inpMrecon = true ;
            MR = val ;
        case {'sininfo'}
            inpsininfo = true;
        case 'geom'
            inpgeom = true ;
            geom = val ;
        case {'labfn'}
            labfn = pref_uigetfile('geomchecker','labfn', val) ;
            if exist(labfn,'file')
              inpMrecon  = true ;
              MR = MRecon(labfn) ;
            end
        case 'msg'
            msg = val ;
            
        otherwise
            error(['Unknown param: ',param])
    end
end
        
% Load file

% % data_path = '/Users/davidatkinson/OneDrive - University College London/data/UNDISTORT/Patient21/documents_20200828' ;
% % fn = 'sp_28032018_1149496_12_2_wip_dwi_0_150_500_1000_orig_nsa6_senseV4.lab' ;



%data_path =  '/Users/davidatkinson/OneDrive - University College London/data/20200922_phantoms/20200922_resphant/datalist' ;
%fn = 'raw_054.data' ;

% ffn = fullfile(data_path, fn) ;

% ffn = '/Users/davidatkinson/OneDrive - University College London/data/20200922_phantoms/20200922_resphant/pngo/2020_09_22/da_347732/da_22092020_1732530_54_2_wip_dwi_a_0V4.lab' ;
% 
% MR = MRecon(ffn) ;

disp(msg)

if inpMrecon
    enc_voxel_sizes = MR.Parameter.GetValue('ENC`ima:voxel_sizes') ;
    enc_ft_lengths =  MR.Parameter.GetValue('ENC`ima:ft_lengths') ;
    enc_sense_factors = MR.Parameter.GetValue('ENC`ima:sense_factors') ;
    enc_profiles = MR.Parameter.GetValue('ENC`ima:profiles') ;
    enc_min_encoding_numbers = MR.Parameter.GetValue('ENC`ima:min_encoding_numbers') ;
    enc_max_encoding_numbers = MR.Parameter.GetValue('ENC`ima:max_encoding_numbers') ;
    enc_oversample_resolutions = MR.Parameter.GetValue('ENC`ima:oversample_resolutions') ;
    
    lca_sampled_fovs = MR.Parameter.GetValue('LCA`ima:[1]:sampled_fovs') ;
    lca_samples =  MR.Parameter.GetValue('LCA`ima:[1]:samples') ;
    
    rc_sense_extra_ovs_factors = MR.Parameter.GetValue('RC_sense_extra_ovs_factors') ;
    rc_recon_resolutions = MR.Parameter.GetValue('RC_recon_resolutions') ;
    rc_oversample_factors = MR.Parameter.GetValue('RC_oversample_factors') ;
    rc_sense_factors = MR.Parameter.GetValue('RC_sense_factors') ;
    rc_min_encoding_numbers = MR.Parameter.GetValue('RC_min_encoding_numbers') ;
    rc_max_encoding_numbers = MR.Parameter.GetValue('RC_max_encoding_numbers') ;
    rc_voxel_sizes = MR.Parameter.GetValue('RC_voxel_sizes') ;
    
    disp( ' ')
    disp(MR.Parameter.Filename)
    disp(['NUS samples ',num2str(MR.Parameter.Labels.NusSamples), ...
        ', [ ', num2str(MR.Parameter.Labels.NusEncNrs(1)), ...
        ' ... ', num2str(MR.Parameter.Labels.NusEncNrs(end)),' ]'])
    
    if length(MR.Parameter.Labels.NusEncNrs) ~= MR.Parameter.Labels.NusSamples
        warning(' NUS lengths not consistent')
    end
    
    % Check RC and ENC encoding numbes are the same (RC are I assume what are
    % in .sin file)
    if ~isequal(rc_min_encoding_numbers(1:3), enc_min_encoding_numbers(1:3))
        warning(['RC and ENC min encoding numbers differ'])
    end
    
    if ~isequal(rc_max_encoding_numbers(1:3), enc_max_encoding_numbers(1:3))
        warning(['RC and ENC max encoding numbers differ'])
    end
    
    
    disp( sprintf('\n enc encoding numbers [%d %d], [%d %d], [%d %d]', ...
        enc_min_encoding_numbers(1) , enc_max_encoding_numbers(1), ...
        enc_min_encoding_numbers(2) , enc_max_encoding_numbers(2), ...
        enc_min_encoding_numbers(3) , enc_max_encoding_numbers(3) ) )
    
    enc_range = enc_max_encoding_numbers - enc_min_encoding_numbers + 1 ;
    disp(['max-min+1 encoding numbers ', num2str(enc_range)])
    
    disp(['enc_profiles ',num2str(enc_profiles)])
    
    if ~isequal(enc_profiles, enc_range)
        warning(['min - max encoding numbers do not correspond to enc_profiles'])
    end
    
    disp(['Actually acquired (after NUS) resolutions (extra sense not in acq)'])
    disp(['enc_oversample_resolutions : ', num2str(enc_oversample_resolutions)])
    disp(['lca_samples (may be out by 1 for EPI) : ',num2str(lca_samples)])
    
    % Fourier Transfor Lengths
    disp(['ENC FT lengths ', num2str(enc_ft_lengths)])
    
    % nkx_zf = osf(1) * recon_res(1) / info.sin.sense_factors(1) ;
    % in above, sense_factors are true_sense * extra_sense
    %           oversample_factors also includes extra_sense)
    % RC_oversample_fctors includes the extra_sense (but ENC oversample factors
    % does not)
    %
    % Likely also true in .LIST for ky_oversample_factor, Y-direction SENSE factor
    %
    
    sinftlength = rc_oversample_factors(1:3) .* rc_recon_resolutions(1:3) ./ ...
        rc_sense_factors ;
    disp([' In kparse FT lengths from sin file expected to be: ', ...
        num2str(sinftlength) ])
    
    disp(['lca_sampled_fovs = ',num2str(lca_sampled_fovs), ' mm'])
    voxs = lca_sampled_fovs./enc_ft_lengths ;
    disp(['voxs : lca_sampled_fovs ./ enc_ft_lengths ',num2str(voxs)])
    
    
    disp(['enc_voxel_sizes              ', num2str(enc_voxel_sizes)])
    disp([' Parameter.Scan.RecVoxelSize ', num2str(MR.Parameter.Scan.RecVoxelSize)])
    disp([' RC_voxel_sizes              ', num2str(rc_voxel_sizes(1:3)) ])
    disp([' IF_recon_voxel_size         ',MR.Parameter.GetValue('IF_recon_voxel_size')])
    
    if abs(voxs(1) - enc_voxel_sizes(1)) > 0.01 || abs(voxs(2) - enc_voxel_sizes(2)) > 0.01
        warning([' Voxel sizes not consistent with enc voxels'])
    end
    if abs(voxs(1) - MR.Parameter.Scan.RecVoxelSize(1)) > 0.01 || abs(voxs(2) - MR.Parameter.Scan.RecVoxelSize(2)) > 0.01
        warning([' Voxel sizes not consistent with MR.Parameter.Scan.RecVoxelSize'])
    end
    
    
    disp('Acq voxels')
    disp([' lca_sampled_fovs ./ enc_profiles ', num2str(lca_sampled_fovs ./ enc_profiles)])
    disp([' Parameter.Scan.AcqVoxelSize      ', num2str(MR.Parameter.Scan.AcqVoxelSize)])
    disp([' IF_meas_voxel_size               ',MR.Parameter.GetValue('IF_meas_voxel_size')])
    
    disp('FOV & SENSE')
    disp(['enc_sense_factors            ', num2str(enc_sense_factors)])
    disp(['rc_sense_extra_ovs_factors : ', num2str(rc_sense_extra_ovs_factors)])
    
    FOV = lca_sampled_fovs .* enc_sense_factors;
    disp(['FOV inc  normal SENSE ', num2str(FOV)])
    
    disp(['FOV after extra SENSE ', num2str(FOV .* rc_sense_extra_ovs_factors ) ])
    
    disp(['FOV recon_res*voxs ',num2str(rc_recon_resolutions(1:3) .* voxs )])
    disp(['Parameter.Labels.FOV_AP_FH_RL = ', num2str(MR.Parameter.Labels.FOV_AP_FH_RL)])
    
    % Offsets etc
    % MPS directions
    % User-specified offsets
    disp([' MPS : ',MR.Parameter.Scan.MPS])
    MPSOffcentresMM = MR.Parameter.Scan.MPSOffcentresMM ;
    disp([' MPSOffcentresMM ',num2str(MPSOffcentresMM)])
    est_vox_offcentres = round(MPSOffcentresMM ./ voxs ) ;
    disp([' MPSOffcentres ', num2str(MR.Parameter.Scan.MPSOffcentres), ...
        ' est_vox_offcentres = ',num2str(est_vox_offcentres)])
    
    disp(['The DC image point will be at isocentre for EPI scans'])
    % check now for shifts (unsymmetric Range) and location offsets
    
    val = MR.Parameter.GetValue('RC_location_spectrum_offsets');
    disp(['First loca RC_location_spectrum_offsets: ', ...
        num2str([val(1) val(257)  val(256*2+1)])])
    
    % Compute geom structure for recon data, apply in plane transforms and
    % compare to Phiips DICOM.
    
    % Compute where MRecon thinks the centre of the midslice is
    midsl = ceil((MR.Parameter.Scan.ImagesPerStack+1)/2) ;
    [xT,A] = MR.Transform([1 1 midsl], 'REC','RAF') ;
    [xT1,A] = MR.Transform([1 2 midsl], 'REC','RAF') ;
    [xT2,A] = MR.Transform([2 1 midsl], 'REC','RAF') ;
    
    disp(['From MRecon, predicted DICOM IPP will be: ',num2str(xT.')])
    disp(['From MRecon, DICOM IOP(1:3) : ',num2str((xT1-xT).'/voxs(2)), ...
        ' IOP(4:6) : ',num2str((xT2-xT).'/voxs(1)) ])
    disp('!! Fix voxs (1 or 2)')
end

if inpgeom
    [slsep, rdc, cdc, sdc] = geom_check(geom, 'opstr', true) ;
end

if inpsininfo
    
end





