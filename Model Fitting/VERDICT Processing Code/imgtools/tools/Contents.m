% TOOLS
%
% Files
%   colorder         - Set color order for lines in plots.
%   DCgp             - Return the DC grid point as a 3D coordinate.
%   defuigetfile     - DEUIGETFILE Similar to uigetfile but uses a stored default directory
%   dgeom            - Converts between DICOM IPP, IOP and offcentres and angulations
%   dgeomextract     - Extract geometry information from dinfo structure
%   dicoman          - DICOM anonymisation keeping only specified information.
%   dinfostrip       - Strip out DICOM info data, keeping only limited tags
%   dreslice         - Reslice data. Investigational use only.
%   gen_dicom_mat    - Generate matrices to convert between DICOM LPS coords and
%   geom_check       - Checks slices are parallel, calculate slice centre separation
%   i2k              - Returns the complex k-space data, given complex image domain.
%   i2ksbs           - I2K Slice by slice
%   k2i              - Returns the complex image data, given complex k-space     
%   k2isbs           - K2I Slice by slice
%   mpr_geom         - Multi-plane Reformat Geometry
%   NaN2zero         - NaN2zero  Sets any pixels with the value NaN to zero.
%   plane_process    - Process a plane structure. Checks or sets plane vectors
%   plot_humidity    - Plots humidity file from Philips monitoring of MR room.
%   plot_temperature - Plot Philips temperature file for monitoring MR room.
%   point_plane      - Point to plane distance
%   pref_uigetdir    - Calls uigetdir with stored user preferences
%   pref_uigetfile   - Calls uigetfile with stored user preferences
%   rotn90           - Rotate n-D matrix by 90 degrees
%   set_plane        - Set a plane structure
%   SpinCalc         - Function for the conversion of one rotation input type to desired output.
