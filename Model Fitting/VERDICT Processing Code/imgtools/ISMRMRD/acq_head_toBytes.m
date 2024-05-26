function bytes = acq_head_toBytes(obj)
% Convert to an ISMRMRD AcquisitionHeader to a byte array
%
% bytes = toBytes(hdr)
%
% hdr is typically ddata.head where ddata is from reading HDF5 file
% bytes is [340 Nprof]
%
% This conforms to the memory layout of the C-struct
%
% Adapted from AcquisitionHeader.m in ISMRMRD github
%
% 


N = length(obj.version) ;
bytes = zeros(340,N,'uint8');
for p = 1:N
    off = 1;
    bytes(off:off+1,p)   = typecast(obj.version(p)               ,'uint8'); off=off+2;
    bytes(off:off+7,p)   = typecast(obj.flags(p)                 ,'uint8'); off=off+8;
    bytes(off:off+3,p)   = typecast(obj.measurement_uid(p)       ,'uint8'); off=off+4;
    bytes(off:off+3,p)   = typecast(obj.scan_counter(p)          ,'uint8'); off=off+4;
    bytes(off:off+3,p)   = typecast(obj.acquisition_time_stamp(p),'uint8'); off=off+4;
    bytes(off:off+11,p)  = typecast(obj.physiology_time_stamp(:,p) ,'uint8'); off=off+12;
    bytes(off:off+1,p)   = typecast(obj.number_of_samples(p)     ,'uint8'); off=off+2;
    bytes(off:off+1,p)   = typecast(obj.available_channels(p)    ,'uint8'); off=off+2;
    bytes(off:off+1,p)   = typecast(obj.active_channels(p)       ,'uint8'); off=off+2;
    bytes(off:off+127,p) = typecast(obj.channel_mask(:,p)        ,'uint8'); off=off+128;
    bytes(off:off+1,p)   = typecast(obj.discard_pre(p)           ,'uint8'); off=off+2;
    bytes(off:off+1,p)   = typecast(obj.discard_post(p)          ,'uint8'); off=off+2;
    bytes(off:off+1,p)   = typecast(obj.center_sample(p)         ,'uint8'); off=off+2;
    bytes(off:off+1,p)   = typecast(obj.encoding_space_ref(p)    ,'uint8'); off=off+2;
    bytes(off:off+1,p)   = typecast(obj.trajectory_dimensions(p) ,'uint8'); off=off+2;
    bytes(off:off+3,p)   = typecast(obj.sample_time_us(p)        ,'uint8'); off=off+4;
    bytes(off:off+11,p)  = typecast(obj.position(:,p)            ,'uint8'); off=off+12;
    bytes(off:off+11,p)  = typecast(obj.read_dir(:,p)            ,'uint8'); off=off+12;
    bytes(off:off+11,p)  = typecast(obj.phase_dir(:,p)           ,'uint8'); off=off+12;
    bytes(off:off+11,p)  = typecast(obj.slice_dir(:,p)           ,'uint8'); off=off+12;
    bytes(off:off+11,p)  = typecast(obj.patient_table_position(:,p),'uint8'); off=off+12;
    bytes(off:off+1,p)   = typecast(obj.idx.kspace_encode_step_1(p),'uint8'); off=off+2;
    bytes(off:off+1,p)   = typecast(obj.idx.kspace_encode_step_2(p),'uint8'); off=off+2;
    bytes(off:off+1,p)   = typecast(obj.idx.average(p)           ,'uint8'); off=off+2;
    bytes(off:off+1,p)   = typecast(obj.idx.slice(p)             ,'uint8'); off=off+2;
    bytes(off:off+1,p)   = typecast(obj.idx.contrast(p)          ,'uint8'); off=off+2;
    bytes(off:off+1,p)   = typecast(obj.idx.phase(p)             ,'uint8'); off=off+2;
    bytes(off:off+1,p)   = typecast(obj.idx.repetition(p)        ,'uint8'); off=off+2;
    bytes(off:off+1,p)   = typecast(obj.idx.set(p)               ,'uint8'); off=off+2;
    bytes(off:off+1,p)   = typecast(obj.idx.segment(p)           ,'uint8'); off=off+2;
    bytes(off:off+15,p)  = typecast(obj.idx.user(:,p)            ,'uint8'); off=off+16;
    bytes(off:off+31,p)  = typecast(obj.user_int(:,p)            ,'uint8'); off=off+32;
    bytes(off:off+31,p)  = typecast(obj.user_float(:,p)          ,'uint8');
end