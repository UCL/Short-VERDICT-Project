function [filename, pathname] = defuigetfile(varargin)
% DEUIGETFILE Similar to uigetfile but uses a stored default directory
%
% ! Note this should be phased out in favour of using setpref and getpref
% See pref_uigetdir.
%
% [filename, pathname] = defuigetfile
% [filename, pathname] = defuigetfile('FilterSpec') relative to def dir
%                         e.g. defuigetfile('*.txt')
% [filename, pathname] = defuigetfile(...)  as uigetfile
%
% The default directory is stored in the global variable DEFDATADIR
%
% David Atkinson, Radiological Sciences, Guy's Hospital, KCL, London
% %W% , created %G%
%
% See also UIGETFILE TOOLS

global MATLAB_HOME DEFDATADIR

if length(DEFDATADIR) == 0
  DEFDATADIR = MATLAB_HOME ;
end


if nargin == 0
  FilterSpec = fullfile(DEFDATADIR,'*') ;
elseif nargin >= 1
  switch class(varargin{1})
    case 'cell'
      warning([ 'defuigetfile cannot cope with cell filterspecs'])
      FilterSpec = fullfile(DEFDATADIR,'*') ;
    otherwise
      FilterSpec = fullfile(DEFDATADIR,varargin{1}) ;
  end
end

[filename, pathname] = uigetfile( FilterSpec, varargin{2:nargin} ) ;
  
if ischar(filename)
  DEFDATADIR = pathname ;
end

