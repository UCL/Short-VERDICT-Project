function [imdata, imhdr] = explore_gclient(varargin)
% explore_gclient. MATLAB Gadgetron client
% Demo client for Gadgetron. Limited features implemented.
% Requires MATLAB Instrument Control Toolbox.
%
%  [imdata, imhdr] = explore_gclient(param, value, ...)
%
% imdata and imhdr are cell arrays containing the reconstructed image(s) 
%  and header data. 
%  For example, if averages of a 2D dataset are returned separately, imdata
%  will have length 5. Data can be put into a 3D matrix: cat(3,imdata{:})
%
% Parameter, value pairs | {Default value}
% 'ipaddr' IP address for remote Gadgetron {'192.168.230.130'}. Use
%          'localhost' for same machine.
% 'port' Port for Gadgetron {9002}
% 'remote_config_file' Configuration File name {'default.xml'}
% 'local_config_file' Configuration File name on client side {''} 
%    For the config file, last param/value pair present will be used. 
%    If neither is present, the default remote_config_file will be used. 
%    If local_config_file is a parameter but the file does not exist, a 
%    GUI will be called to select the file.
% 'ismrmrd_file' ISMRMRD file name {''} blank calls GUI
%
% Example usage:
%  % Start Gadgetron on remote machine (here with IP 192.168.230.130).
%  
% If using Docker, allow port eg.
%  docker run --name=gtclass --publish=9002:9002 --publish=9080:9080 --publish=8090:8090 --volume=/Users/davidatkinson/gadgetron-course:/opt/data  --detach gadgetron/ubuntu_2004
% 
%  optionally open terminal and tail -f
%   docker exec -ti gtclass /bin/bash
%  
% David Atkinson  D.Atkinson@ucl.ac.uk
% See ISMRMRD and Gadgetron code on GitHub
%
% See also freadbuf  im_head_fromBytes  read_img_messg  pref_uigetfile acq_head_toBytes
%

% Default Parameter values
ipaddr = '192.168.230.130' ;
port = 9002 ;
remote_config_file = 'default.xml' ;
local_config_file = '' ;
ismrmrd_file = '' ;

config_source = 'remote' ; % default if no config file in input params

% Gadgetron message enumeration. See Gadgetron Python Client code
% https://github.com/gadgetron/gadgetron-python-ismrmrd-client/blob/master/gtconnector.py

GADGET_MESSAGE_CONFIG_FILE = 1 ;
GADGET_MESSAGE_CONFIG_SCRIPT  =  2 ;
GADGET_MESSAGE_PARAMETER_SCRIPT =   3 ;
GADGET_MESSAGE_CLOSE       = 4 ;
GADGET_MESSAGE_ISMRMRD_ACQUISITION = 1008 ;


for ipv = 1: 2: nargin
    param = varargin{ipv} ; val = varargin{ipv+1} ;
    switch param
        % recognised parameter assigned to same-name variable
        case {'port', 'ipaddr', 'ismrmrd_file' }
            eval([param, ' = val ;']) ;
        case 'remote_config_file'
            config_source = 'remote' ;
            remote_config_file = val ;
        case 'local_config_file'
            config_source = 'local' ;
            local_config_file = val ;
        otherwise
            warning(['Parameter: ',param,' not recognised/'])
    end
end

% if exist('t','class')
%    if isa(t,'tcpip')
%      fclose(t);
%      delete(t) ;
%    end
% end

% enum ISMRMRD_DataTypes {
%     ISMRMRD_USHORT   = 1, /**< corresponds to uint16_t */
%     ISMRMRD_SHORT    = 2, /**< corresponds to int16_t */
%     ISMRMRD_UINT     = 3, /**< corresponds to uint32_t */
%     ISMRMRD_INT      = 4, /**< corresponds to int32_t */
%     ISMRMRD_FLOAT    = 5, /**< corresponds to float */
%     ISMRMRD_DOUBLE   = 6, /**< corresponds to double */
%     ISMRMRD_CXFLOAT  = 7, /**< corresponds to complex float */
%     ISMRMRD_CXDOUBLE = 8  /**< corresponds to complex double */
% };

% From IsmrmrdHeader.m
% IMAGE_TYPE = struct( ...
%             'MAGNITUDE', uint16(1), ...
%             'PHASE',     uint16(2), ...
%             'REAL',      uint16(3), ...
%             'IMAG',      uint16(4), ...
%             'COMPLEX',   uint16(5));
        
        
        


%----------------------------
% Prepare input (string from XML part of ISMRMRD)
xml_fn = pref_uigetfile('explore_ismrmrd', 'filename', ismrmrd_file) ;
disp(['Using ISMRMRD file: ',xml_fn])

dxml = h5read(xml_fn,'/dataset/xml') ;
ddata = h5read(xml_fn,'/dataset/data') ;

str_xml = dxml{1} ;


% TCPIP connection to gadgetron.
t = tcpip(ipaddr, port) ;
t.OutputBufferSize = 512*512 ;
t.InputBufferSize = 512*512*8 ; % arbitrary choice - may need attention
t.ByteOrder = 'littleEndian' ; 

fopen(t)

% Configuration File

switch config_source
    case 'remote'
        % Config file on Gadgetron (remote) machine
        config_file_withterm = [uint8(remote_config_file) 0 10] ; % add null and LF
        cstr = zeros([1 1024]) ; % place in array of length 1024
        cstr(1:length(config_file_withterm)) = config_file_withterm ;
        
        fwrite(t, GADGET_MESSAGE_CONFIG_FILE,'uint16') ;
        fwrite(t,cstr,'uint8') ; % Send string within the 1024 bytes
    case 'local'
        % Config is xml file on local (client) machine - send as script
        config_local_fn = pref_uigetfile('explore_ismrmrd', ...
                                'local_config_fn', local_config_file) ;
        
        % read xml config file into string.
        fid_cf = fopen(config_local_fn) ;
        [c_xml_str] = fscanf(fid_cf,'%c') ;
        fclose(fid_cf) ;
        uc_xml_str_withterm = [uint8(c_xml_str) 0 10] ;
        
        fwrite(t, GADGET_MESSAGE_CONFIG_SCRIPT ,'uint16') ;
        fwrite(t, length(uc_xml_str_withterm),'uint32') ; % Write length
        fwrite(t,uc_xml_str_withterm,'uint8') ; % Send xml string
        
        
    otherwise
        error(['Unknown config_source: ',config_source])
end

% Parameter message  (XML from ISMRMRD HDF5 file)
% Create xml string 
ustr = [uint8(str_xml) 0 10] ; % null + LF
len_xml = length(ustr) ; 

fwrite(t, GADGET_MESSAGE_PARAMETER_SCRIPT, 'uint16')
fwrite(t, len_xml,'uint32') ; % Write length
fwrite(t, ustr, 'uint8') ;  % Write XML

% send_ismrmrd_acquisition 
% Loops round the profiles sending [id][acq header][traj][data]
acq_h_bytes = acq_head_toBytes(ddata.head) ; 

nprof = size(acq_h_bytes,2) ;
for iprof = 1:nprof
    fwrite(t, GADGET_MESSAGE_ISMRMRD_ACQUISITION, 'uint16') ;
    fwrite(t, acq_h_bytes(:,iprof), 'uint8') ;
    
    trajectory_elements = ddata.head.trajectory_dimensions(iprof) * ...
        ddata.head.number_of_samples(iprof);
    
    data_elements = ddata.head.active_channels(iprof) * ...
        ddata.head.number_of_samples(iprof);
    
    if trajectory_elements > 0
        fwrite(t, ddata.traj{iprof}, 'float') ; %untested
    end
    
    if data_elements > 0
        dr = ddata.data{iprof} ;
        fwrite(t, dr, 'float') ; %assume correct?
    end
end % prof

% Read response from Gadgetron, filling cell arrays for image and header
imc = 1 ; % image counter
while t.BytesAvailable > 0
    [msg, this_imhdr, this_imdata] = read_img_messg(t) ;
    imhdr{imc} = this_imhdr ;
    imdata{imc} = this_imdata ;
    
    imc = imc+1;
end

disp(['Read in ',num2str(imc-1),' images.'])

disp(['Sending close message to: ',ipaddr])

% Send Close message 
fwrite(t, GADGET_MESSAGE_CLOSE, 'uint16')
pause(1)

% Append any further images to the previous
while t.BytesAvailable > 198
    [msg, this_imhdr, this_imdata] = read_img_messg(t) ;
    imhdr{imc} = this_imhdr ;
    imdata{imc} = this_imdata ;
    
    imc = imc+1;
end

% Check for close from Gadgetron
msg = fread(t,1,'uint16');
if msg == GADGET_MESSAGE_CLOSE
    disp(['Close message received from Gadgetron.'])
else
    disp(['Unknown message: ',num2str(msg),'received at end'])
end

if exist('imdata','var')
    disp(['Total images read: ',num2str(length(imdata))])
    disp(['  To display, try:  eshow(cat(3,imdata{:}))'])
else
    disp(['No output imdata'])
end

% Close Gadgetron connection
fclose(t) ;
delete(t)
clear t





        







