function set_external_amri_path(external_dir)
% set_external_amri_path
%

amri_folder = fullfile(external_dir, 'video_phase_processing') ;

p = genpath(amri_folder) ;
addpath(p)
