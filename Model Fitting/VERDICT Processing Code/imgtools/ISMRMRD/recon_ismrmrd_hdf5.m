function h5_fn = recon_ismrmrd_hdf5(varargin)
% RECON_ISMRMRD_HDF5 
% Reads an HDF5 ISMRMRD file, performs reconstruction and places images
% back in this file.
%
%  h5_fn = recon_ismrmrd_hdf5
%  h5_fn = recon_ismrmrd_hdf5(h5_fn)
%
% Recon data can be accessed from file with:
%    dat = hdf5read(h5_fn,'dataset/matlab') ;
%
% For this code to work, you must add <ISMRMRD_SOURCE>/matlab to your path
% Typically this is a folder named matlab containing a subfolder +ismrmrd 
% and further subfolders. Note you need to preserve the naming and structure 
% of these folders for MATLAB to handle the classes correctly.
%
% Scanner test data can be downloaded from:
%      https://github.com/ismrmrd/ismrmrd/releases/download/v1.2.3-data/ismrmrd_data.zip
%      'bruker.h5', 'ge.h5', 'philips.h5', 'siemens.h5'
%
% or synthetic data generated using C++ program:
%    ismrmrd_generate_cartesian_shepp_logan -m 512 -o synth.h5
%
% Based on do_recon_matlab

h5_fn = pref_uigetfile('hdf5_files', 'recon', varargin{:}) ;

disp(['Prior to recon HDF5 file contains:'])
h5disp(h5_fn,'/','min')

recon_dataset(h5_fn) ;

disp(['After recon HDF5 file contains:'])
h5disp(h5_fn,'/','min')

end
%------------%

function recon_dataset(filename)

% This is a simple example of how to reconstruct images from data
% acquired on a fully sampled cartesian grid
%
% Capabilities:
%   2D/3D
%   multiple slices/slabs
%   multiple contrasts, repetitions
%
% Limitations:
%   only works with a single encoded space
%   fully sampled k-space (no partial fourier or undersampling)
%   multiple repetitions
%   doesn't handle averages, phases, segments and sets
%   ignores noise scans (no pre-whitening)
%

% We first create a data set using the example program like this:
%   ismrmrd_generate_cartesian_shepp_logan -r 5 -C -o shepp-logan.h5
% This will produce the file shepp-logan.h5 containing an ISMRMRD
% dataset sampled evenly on the k-space grid -128:0.5:127.5 x -128:127
% (i.e. oversampled by a factor of 2 in the readout direction)
% with 8 coils, 5 repetitions and a noise level of 0.5
% with a noise calibration scan at the beginning
%
%

%%%%%%%%%%%%%%%%%%%%
% Loading the file %
%%%%%%%%%%%%%%%%%%%%
if exist(filename, 'file')
    dset = ismrmrd.Dataset(filename, 'dataset');
else
    error(['File ' filename ' does not exist.  Please generate it.'])
end

h5i = h5info(filename) ; % used later to check for exising Dataset/matlab

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Read some fields from the XML header %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% We need to check if optional fields exists before trying to read them

hdr = ismrmrd.xml.deserialize(dset.readxml);

%% Encoding and reconstruction information
% Matrix size
enc_Nx = hdr.encoding.encodedSpace.matrixSize.x;
enc_Ny = hdr.encoding.encodedSpace.matrixSize.y;
enc_Nz = hdr.encoding.encodedSpace.matrixSize.z;
rec_Nx = hdr.encoding.reconSpace.matrixSize.x;
rec_Ny = hdr.encoding.reconSpace.matrixSize.y;
rec_Nz = hdr.encoding.reconSpace.matrixSize.z;

% Field of View
enc_FOVx = hdr.encoding.encodedSpace.fieldOfView_mm.x;
enc_FOVy = hdr.encoding.encodedSpace.fieldOfView_mm.y;
enc_FOVz = hdr.encoding.encodedSpace.fieldOfView_mm.z;
rec_FOVx = hdr.encoding.reconSpace.fieldOfView_mm.x;
rec_FOVy = hdr.encoding.reconSpace.fieldOfView_mm.y;
rec_FOVz = hdr.encoding.reconSpace.fieldOfView_mm.z;

% Number of slices, coils, repetitions, contrasts etc.
% We have to wrap the following in a try/catch because a valid xml header may
% not have an entry for some of the parameters

try
  nSlices = hdr.encoding.encodingLimits.slice.maximum + 1;
catch
    nSlices = 1;
end

try
    nCoils = hdr.acquisitionSystemInformation.receiverChannels;
catch
    nCoils = 1;
end

try
    nReps = hdr.encoding.encodingLimits.repetition.maximum + 1;
catch
    nReps = 1;
end

try
    nContrasts = hdr.encoding.encodingLimits.contrast.maximum + 1;
catch
    nContrasts = 1;
end

%% Read all the datas
% Reading can be done one acquisition (or chunk) at a time,
% but this is much faster for data sets that fit into RAM.
D = dset.readAcquisition();

% Ignore noise scans
% Find the first non-noise scan
% This is how to check if a flag is set in the acquisition header
isNoise = D.head.flagIsSet('ACQ_IS_NOISE_MEASUREMENT');
firstScan = find(isNoise==0,1,'first');
if firstScan > 1
    noise = D.select(1:firstScan-1);
else
    noise = [];
end

% Noise display
noise_lines = find(isNoise==1) ;
noise_data = D.select(noise_lines) ;
noise_k = noise_data.data{1} ;  
[np, nc] = size(noise_k) ;
noise_kp = permute(noise_k,[2 1]) ;
scale_factor = 1 ; % assumed here that dwell and bandwidth same for noise as data
ncvm = 1/(np-1) * (noise_kp * noise_kp') ;
eshow(ncvm,'Name','Noise CoV matrix')

dmtx = ismrm_calculate_noise_decorrelation_mtx(noise_k, scale_factor) ;
eshow(dmtx,'Name','decorr mtx')


%DA: 
if isempty(firstScan)
    disp(['No non-noise scan detected, assume firstScan is 1'])
    firstScan = 1 ;
end

meas  = D.select(firstScan:D.getNumber);
uquser = unique(meas.head.idx.user','rows') ;
if size(uquser,1) > 1
    warning(['More than one user entry,using first'])
end
        
clear D;

%% Reconstruct images
% Since the entire file is in memory we can use random access
% Loop over repetitions, contrasts, slices
reconImages = zeros(rec_Nx, rec_Ny, nSlices, nContrasts, nReps,'single');

for rep = 1:nReps
    for contrast = 1:nContrasts
        for slice = 1:nSlices
            % Initialize the K-space storage array
            K = zeros(enc_Nx, enc_Ny, enc_Nz, nCoils);
            % Select the appropriate measurements from the data
            acqs = find(  (meas.head.idx.contrast==(contrast-1)) ...
                        & (meas.head.idx.repetition==(rep-1)) ...
                        & (meas.head.idx.slice==(slice-1)) ...
                        & (ismember(meas.head.idx.user', uquser(1,:),'rows')') ...  % DA was uquser(1,:) - why??
                        & (meas.head.idx.average==0) );  % DA
            for p = 1:length(acqs)
                ky = meas.head.idx.kspace_encode_step_1(acqs(p)) + 1;
                kz = meas.head.idx.kspace_encode_step_2(acqs(p)) + 1;
                K(:,ky,kz,:) = meas.data{acqs(p)};
            end
            % Reconstruct in x
            K = fftshift(ifft(fftshift(K,1),[],1),1);
            % Chop if needed
            if (enc_Nx == rec_Nx)
                im = K;
            else
                ind1 = floor((enc_Nx - rec_Nx)/2)+1;
                ind2 = floor((enc_Nx - rec_Nx)/2)+rec_Nx;
                im = K(ind1:ind2,:,:,:);
            end
            % Reconstruct in y then z
            im = fftshift(ifft(fftshift(im,2),[],2),2);
            if size(im,3)>1
                im = fftshift(ifft(fftshift(im,3),[],3),3);
            end

            % Combine SOS across coils
            im = sqrt(sum(abs(im).^2,4));

            % Stuff
            reconImages(:,:,slice,contrast,rep) = single(im);
        end
    end
end

% Append the block to the ismrmrd file as an NDArray

% Check if Dataset/matlab is already in HDF5 file, if so do not create.
GDN = {h5i.Groups.Datasets.Name} ;
tf = strcmp('matlab',GDN) ;
if sum(tf)==0
    h5create(filename,'/dataset/matlab',size(reconImages),'Datatype','single')
else
    disp(['HDF5 file already contains a Dataset called matlab'])
end

h5write(filename,'/dataset/matlab',reconImages)
disp(['Written HDF5 file: ',filename]) 

end
