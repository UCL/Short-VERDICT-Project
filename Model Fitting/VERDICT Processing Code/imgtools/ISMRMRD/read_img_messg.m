function [msg, imhdr, imdata] = read_img_messg(t)
% READ_IMG_MESSG  Reads incoming image message from Gadgetron
% 
%
% [msg, imhdr, imdata] = read_img_messg(t)
%  Outputs are cell arrays.
%
% See ISMRMRD ImageHeader.m
%
% David Atkinson  D.Atkinson@ucl.ac.uk
%
% See also im_head_fromBytes 

IM_HDR_BYTES = 198 ;

tba = t.BytesAvailable ;
if tba < IM_HDR_BYTES
    warning(['Not enough bytes available (',num2str(tba),') to read message and header.'])
end

msg = fread(t,1,'uint16');
if msg ~= 1022
    warning(['Expected message 1022, got: ',num2str(msg)])
else
    % disp(['Read message 1022.'])
end

imhb = fread(t, IM_HDR_BYTES, 'uint8') ;
imhdr = im_head_fromBytes(uint8(imhb)) ;

if imhdr.attribute_string_len > 0
    warning(['Only zero length Attribute string currently read.'])
end

% disp(['Afer reading img header, BytesAvailable are: ',num2str(t.BytesAvailable)])

% Read Attribute String Length (likely to change in future releases)
attr_lena = fread(t,1,'uint32') ;  % overloaded fread does not read 'uint64'
attr_lenb = fread(t,1,'uint32') ;

if attr_lena ~=0 || attr_lenb ~=0
    warning(['Cant handle non zero atrr_lena/b'])
end
attr_len = 0 ;
if attr_len ~= imhdr.attribute_string_len
    warning(['Header and other attibute stream lengths do not match'])
end

% Attribute read will go here

% Read image data
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
switch imhdr.data_type
    case 1 % USHORT
        [imdata, count] = fread(t, prod(imhdr.matrix_size), 'uint16' ) ;
    case 2 % SHORT
        [imdata, count] = fread(t, prod(imhdr.matrix_size), 'int16' ) ;
    case 3 % UINT
        [imdata, count] = fread(t, prod(imhdr.matrix_size), 'uint32' ) ;    
    case 5 % FLOAT
        [imdata, count] = fread(t, prod(imhdr.matrix_size), 'float' ) ;
    otherwise
        warning(['Data type: ',num2str(imhdr.data_type),' not implemented'])
end
%disp(['Read ',num2str(count),' values.'])
%disp(['BytesAvailable are: ',num2str(t.BytesAvailable)])


switch imhdr.image_type
    case 1 % Magnitude
    otherwise
        warning(['Image type not implemented.'])            
end
            
imdata = reshape(imdata,[imhdr.matrix_size(1) imhdr.matrix_size(2) imhdr.matrix_size(3)]) ;
