function UPDOWN_set_path( external_dir )
% UPDWON_set_path( external_dir )
%
% Adds folders to path for UP DOWN code
% If already on path, will move it up.

%MRecon  Usman used MRecon-3.0.552
add_extern(external_dir , 'MRECON','MRecon-4.0.5' ) ;

add_extern(external_dir , 'NIfTI', 'NIfTI_20140122') ;

add_extern(external_dir , 'MEDI_toolbox', 'MEDI_toolbox-devel' ) ;
MEDI_set_path ;

add_extern(external_dir , 'dst_idst' ) ;

% Fessler Code
add_extern(external_dir , 'Fessler', 'irt', 'nufft') ;
add_extern(external_dir , 'Fessler', 'irt', 'nufft', 'table') ;
add_extern(external_dir , 'Fessler', 'irt', 'systems') ;
add_extern(external_dir , 'Fessler', 'irt', 'penalty') ;
add_extern(external_dir , 'Fessler', 'irt', 'general') ;
add_extern(external_dir , 'Fessler', 'irt', 'mri') ;
add_extern(external_dir , 'Fessler', 'irt', 'wls') ;
add_extern(external_dir , 'Fessler', 'irt', 'utilities') ;
add_extern(external_dir , 'Fessler', 'irt', 'fbp') ;
add_extern(external_dir , 'Fessler', 'irt', 'mex') ;
add_extern(external_dir , 'Fessler', 'irt', 'mex' , 'v7') ; % needs to come after mri 

% code2
% NOTE this adds all subfolders AND needs to come after Fessler is added to
% path so that functions are picked up from here (I think)
bpath = fullfile(external_dir, 'code2') ;
if ~exist(bpath,'dir'), warning(['Did not find code2 path: ',bpath]), end
p=genpath(bpath);
addpath(p) ;






function add_extern( edir, varargin)

npath = fullfile( edir, varargin{:} ) ;

if ~exist(npath,'dir')
    warning(['Path not found: ',npath])
else
    addpath( npath ) ;
end