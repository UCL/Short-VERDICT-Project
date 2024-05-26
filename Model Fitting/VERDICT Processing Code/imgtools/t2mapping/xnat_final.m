function xnat_final(InputFolder,OutputFolder,f_name)
% XNAT_FINAL  
%
% xnat_final(InputFolder,OutputFolder,f_name)
%
% Code from Will Devine (previous extension was .mat)
%

%% NOTE
% 1. Each of the 'Item_'s in 'dat1.PerFrameFunctionalGroupsSequence' should 
% correspond to a slice in the image. There are too many for the map as the
% dicom info is taken from the ME-T2 image which has 32*6 slices rather
% than 6 actual slices.

% 2. There are a lot of values in the new dicoms that are not correct. The
% dicom info is taken from the original 32-echo T2 images so the slice
% numbers are wrong, the locations are wrong, the T2 values are not
% applicable etc. Many of these need changing/removing so the best thing
% would be to create a new dicominfo and only copy across the factors we
% want, leaving behind the incorrect values.



cd(InputFolder)

d = dir;
f ={d.name};
f_index = sum(contains(f,'.dcm'));

if f_index == 0
    error(strcat('No dicom files found in folder: ',InputFolder))
elseif f_index > 1
%     error(strcat('More than one dicom file found in folder: ',InputFolder))
end

filename = f{contains(f,'.dcm')};
  
dat = dicominfo(filename);
nEchoes = dat.EchoTrainLength;
TE = (1/1000)*dat.PerFrameFunctionalGroupsSequence.Item_1.MREchoSequence.Item_1.EffectiveEchoTime;
T2 = TE.*(1:nEchoes);

raw_image = squeeze(double(dicomread(filename)));
IMAGE = permute(reshape(raw_image,[size(raw_image,1),size(raw_image,2),nEchoes,size(raw_image,3)./nEchoes]),[1,2,4,3]);

%% set initial parameters
iter = 1000;                                                                % number of 'exponentials' that we are going to fit to the signal decay
max_iter = 2000e-3;                                                         % (in seconds) highest T2 value that the exponentials will fit to
t2_iter = linspace(0,max_iter,iter);                                        % create a vector of T2 values, each of which will have an exponential to fit to the data
s_init = 5e-3;                                                              % initial standard deviation parameter for each of the two Gaussian ppeaks that we are going to fit (this value seems to work and is about right)
s_max = 1e-2;                                                               % maximum value of the stand. devs. of Gaussian peaks, unusual to reach this but it stops code running for a long time on pixels outside prostate
bound = 200e-3;                                                             % (in seconds) boundary between the short peak and the long peak (based on experimental and theoretical values, nothing under should be lumen and nothing above epithelia or stroma)
%% set parameters
x = size(IMAGE,1);
y = size(IMAGE,2);
z = size(IMAGE,3);

S = nan([x,y,z,6]);                                                         % set up empty matrix for the outputs of the two-gaussian fitting (4th dimension is 6 parameters of the model)
C = nan([x,y,z,numel(T2)]);                                                 % set up empty matrix for the estimated signal decay based upon the parameters in S
P = nan([x,y,z,iter]);                                                      % set up empty matrix for the whole T2 spectrum based upon the parameters in S
P_short = nan([x,y,z,size(t2_iter,2)]);                                     % set up empty matrix for the short peak of the T2 spectrum only, based upon the parameters in S
P_long = nan([x,y,z,size(t2_iter,2)]);                                      % set up empty matrix for the long peak of the T2 spectrum only, based upon the parameters in S

%% find initial estimates using bi-exponential fit of the median signal of each slice and echo
F = nan([x,y,z,4]);                                                         % set up empty matrix for the initial estimates
sig_med = nan([size(IMAGE,3),size(IMAGE,4)]);                               % set up empty matrix for the median signal of a slice and echo

fo = fitoptions('exp2','Lower',[0,-inf,0,-inf],'Upper',[inf,-1/max_iter,inf,-1/max_iter],'display','off'); % set up the fit including max and min values
    
for k = 1:size(IMAGE,3)                                                     % loop through slices
    for e = 1:size(IMAGE,4)                                                 % loop through echoes
        M = IMAGE(:,:,k,e);                                                 % set M as the image of echo e in slice k
        sig_med(k,e) = nanmedian(M(:));                                     % take the median of M, excluding any NaN values
    end
    if max(~isnan(sig_med(k,:)))==1                                         % if in sig_med there are no NaN values for any of the echoes in slice k
        f = fit(T2',sig_med(k,:)','exp2',fo);                               % do the fitting for slice k
        F(:,:,k,:) = permute(repmat([f.a,f.b,f.c,f.d],[x,1,y]),[1,3,2]);    % feed this information into F in the correct format
    else
        F(:,:,k,:) = nan;
    end
end

%% initialise pre-LWI fit
X0 = nan([x,y,z,6]);                                                        % set up matrix for initial values for the pre-LWI fit 
X0(:,:,:,1) = F(:,:,:,1)+F(:,:,:,3)./100;                                   % initialise overall magnitude M0 (take average of bi-exponential magnitudes and divide by 100, works as initial value)
X0(:,:,:,2) = F(:,:,:,1)./(F(:,:,:,1)+F(:,:,:,3));                          % initialise relative gaussian peak magnitudes alpha
X0(:,:,:,3) = -1.*ones(size(F(:,:,:,2)))./F(:,:,:,2);                       % initialise mean of short gaussian peak
X0(:,:,:,4) = -1.*ones(size(F(:,:,:,4)))./F(:,:,:,4);                       % initialise mean of long gaussian peak
X0(:,:,:,5) = s_init;                                                       % initialise stand. dev. of short gaussian peak
X0(:,:,:,6) = s_init;                                                       % initialise stand. dev. of long gaussian peak

X0(X0(:,:,:,3)>bound)=bound;                                                % if any of the initial estimates for the short mean are larger than the bound, set them to the same value as the bound
X0(X0(:,:,:,4)<bound)=bound;                                                % if any of the initial estimates for the long mean are smaller than the bound, set them to the same value as the bound

lb = [0,0,0,bound,0,0];                                                     % set lower bounds for each parameter in the LWI fit
ub = [inf,1,bound,max_iter,s_max,s_max];                                    % set upper bounds for each parameter in the LWI fit
pr = @(s) s(1).*((s(2).*normpdf(t2_iter',s(3),s(5)))+((1-s(2)).*normpdf(t2_iter',s(4),s(6)))); % set up the shape of the fitting, if we use 'pr(s)' we will get a spectrum of T2 distributions that is made up of two gaussians with parameters 's'
A = exp(-T2'*(1./t2_iter));

%% execute pre-LWI fit
F2 = nan([x,y,z,6]);
for k = 1:z
    if max(isnan(X0(1,1,k,:)))
        continue
    else
        fun = @(s) (A*pr(s))-squeeze(sig_med(k,:)');                        % define the function that we want to minimise, the difference between the signal we model and the actual signal
        F2(:,:,k,:) = permute(repmat(lsqnonlin(fun,squeeze(X0(1,1,k,:)),lb,ub),[1,x,y]),[2,3,4,1]); % carry out pre-LWI two-gaussian fit
    end
end

%% execute LWI fit
for i = 1:x
    for j = 1:y
        for k = 1:z
            if max(isnan(IMAGE(i,j,k,:)))
                continue
            else
                fun = @(s) (A*pr(s))-squeeze(IMAGE(i,j,k,:));               % set up LWI fit on pixel (x,y,z)
                S(i,j,k,:) = lsqnonlin(fun,F2(i,j,k,:),lb,ub);              % carry out LWI fit
                C(i,j,k,:) = A*pr(S(i,j,k,:));                              % calculate the signal decay estimatd by the model
                P(i,j,k,:) = pr(S(i,j,k,:));                                % calculate the spectrum estimated by the model
                P_short(i,j,k,:) = normpdf(t2_iter',S(i,j,k,3),S(i,j,k,5)); % calculate the spectrum of the short peak estimated by the model
                P_long(i,j,k,:) = normpdf(t2_iter',S(i,j,k,4),S(i,j,k,6));  % calculate the spectrum of the short peak estimated by the model
            end
        end
    end
end
A1 = trapz(t2_iter,repmat(S(:,:,:,1).*S(:,:,:,2),[1,1,1,size(P_short,4)]).*P_short,4); % calculate the area of the short peaks
A2 = trapz(t2_iter,repmat(S(:,:,:,1).*(ones(size(S(:,:,:,2)))-S(:,:,:,2)),[1,1,1,size(P_long,4)]).*P_long,4); % calculate the area of the long peaks
LWF = A2./(A1+A2);                                                          % calculate the LWF
O = cat(4,S,A1,A2,LWF);
%% save dicom and nifti in 
cd(OutputFolder)
mkdir(f_name)
cd(f_name)

names = {'M0','alpha','mean_1','mean_2','sd_1','sd_2','A1','A2','LWF'};
for i=1:size(O,4)
    dat1 = dat;
    dat1.SeriesDescription = names{i};
    dat1.SeriesNumber = double(strcat('240',string(i)));
    dat1.RescaleSlope = max(max(max(O(:,:,:,i))))./1000;
    dat1.RescaleIntercept=0;
    for j=1:size(O,3)
        item_no = char(strcat('Item_',string(j)));
        item_no2 = char(strcat('Item_',string((j*32)-31)));
        dat1.PerFrameFunctionalGroupsSequence.(item_no).PixelValueTransformationSequence.Item_1.RescaleSlope = dat1.RescaleSlope;
        dat1.PerFrameFunctionalGroupsSequence.(item_no).PixelValueTransformationSequence.Item_1.RescaleIntercept = dat1.RescaleIntercept;
        dat.PerFrameFunctionalGroupsSequence.(item_no).PlanePositionSequence.Item_1.ImagePositionPatient(3) = dat.PerFrameFunctionalGroupsSequence.(item_no2).PlanePositionSequence.Item_1.ImagePositionPatient(3);  
    end
    dicomwrite(uint16(permute(O(:,:,:,i),[1,2,4,3]).*1000./max(max(max(O(:,:,:,i))))),strcat(names{i},'.dcm'),dat1,'CreateMode','copy')

    n = make_nii(O);
    save_nii(n,strcat(names{i},'.nii'))
end









%% FUNCTION TO MAKE A NIFTI
function nii = make_nii(varargin)

   nii.img = varargin{1};
   dims = size(nii.img);
   dims = [length(dims) dims ones(1,8)];
   dims = dims(1:8);

   voxel_size = [0 ones(1,7)];
   origin = zeros(1,5);
   descrip = '';

   switch class(nii.img)
      case 'uint8'
         datatype = 2;
      case 'int16'
         datatype = 4;
      case 'int32'
         datatype = 8;
      case 'single'
         if isreal(nii.img)
            datatype = 16;
         else
            datatype = 32;
         end
      case 'double'
         if isreal(nii.img)
            datatype = 64;
         else
            datatype = 1792;
         end
      case 'int8'
         datatype = 256;
      case 'uint16'
         datatype = 512;
      case 'uint32'
         datatype = 768;
      otherwise
         error('Datatype is not supported by make_nii.');
   end

   if nargin > 1 & ~isempty(varargin{2})
      voxel_size(2:4) = double(varargin{2});
   end

   if nargin > 2 & ~isempty(varargin{3})
      origin(1:3) = double(varargin{3});
   end

   if nargin > 3 & ~isempty(varargin{4})
      datatype = double(varargin{4});

      if datatype == 128 | datatype == 511
         dims(5) = [];
         dims(1) = dims(1) - 1;
         dims = [dims 1];
      end
   end

   if nargin > 4 & ~isempty(varargin{5})
      descrip = varargin{5};
   end

   if ndims(nii.img) > 7
      error('NIfTI only allows a maximum of 7 Dimension matrix.');
   end

   maxval = round(double(max(nii.img(:))));
   minval = round(double(min(nii.img(:))));

   nii.hdr = make_header(dims, voxel_size, origin, datatype, ...
	descrip, maxval, minval);

   switch nii.hdr.dime.datatype
   case 2
      nii.img = uint8(nii.img);
   case 4
      nii.img = int16(nii.img);
   case 8
      nii.img = int32(nii.img);
   case 16
      nii.img = single(nii.img);
   case 32
      nii.img = single(nii.img);
   case 64
      nii.img = double(nii.img);
   case 128
      nii.img = uint8(nii.img);
   case 256
      nii.img = int8(nii.img);
   case 511
      img = double(nii.img(:));
      img = single((img - min(img))/(max(img) - min(img)));
      nii.img = reshape(img, size(nii.img));
      nii.hdr.dime.glmax = double(max(img));
      nii.hdr.dime.glmin = double(min(img));
   case 512
      nii.img = uint16(nii.img);
   case 768
      nii.img = uint32(nii.img);
   case 1792
      nii.img = double(nii.img);
   otherwise
      error('Datatype is not supported by make_nii.');
   end

   return;					% make_nii


%---------------------------------------------------------------------
function hdr = make_header(dims, voxel_size, origin, datatype, ...
	descrip, maxval, minval)

   hdr.hk   = header_key;
   hdr.dime = image_dimension(dims, voxel_size, datatype, maxval, minval);
   hdr.hist = data_history(origin, descrip);
    
   return;					% make_header


%---------------------------------------------------------------------
function hk = header_key

    hk.sizeof_hdr       = 348;			% must be 348!
    hk.data_type        = '';
    hk.db_name          = '';
    hk.extents          = 0;
    hk.session_error    = 0;
    hk.regular          = 'r';
    hk.dim_info         = 0;
    
    return;					% header_key


%---------------------------------------------------------------------
function dime = image_dimension(dims, voxel_size, datatype, maxval, minval)
   
   dime.dim = dims;
   dime.intent_p1 = 0;
   dime.intent_p2 = 0;
   dime.intent_p3 = 0;
   dime.intent_code = 0;
   dime.datatype = datatype;
   
   switch dime.datatype
   case 2,
      dime.bitpix = 8;  precision = 'uint8';
   case 4,
      dime.bitpix = 16; precision = 'int16';
   case 8,
      dime.bitpix = 32; precision = 'int32';
   case 16,
      dime.bitpix = 32; precision = 'float32';
   case 32,
      dime.bitpix = 64; precision = 'float32';
   case 64,
      dime.bitpix = 64; precision = 'float64';
   case 128
      dime.bitpix = 24;  precision = 'uint8';
   case 256 
      dime.bitpix = 8;  precision = 'int8';
   case 511
      dime.bitpix = 96;  precision = 'float32';
   case 512 
      dime.bitpix = 16; precision = 'uint16';
   case 768 
      dime.bitpix = 32; precision = 'uint32';
   case 1792,
      dime.bitpix = 128; precision = 'float64';
   otherwise
      error('Datatype is not supported by make_nii.');
   end
   
   dime.slice_start = 0;
   dime.pixdim = voxel_size;
   dime.vox_offset = 0;
   dime.scl_slope = 0;
   dime.scl_inter = 0;
   dime.slice_end = 0;
   dime.slice_code = 0;
   dime.xyzt_units = 0;
   dime.cal_max = 0;
   dime.cal_min = 0;
   dime.slice_duration = 0;
   dime.toffset = 0;
   dime.glmax = maxval;
   dime.glmin = minval;
   
   return;					% image_dimension


%---------------------------------------------------------------------
function hist = data_history(origin, descrip)
   
   hist.descrip = descrip;
   hist.aux_file = 'none';
   hist.qform_code = 0;
   hist.sform_code = 0;
   hist.quatern_b = 0;
   hist.quatern_c = 0;
   hist.quatern_d = 0;
   hist.qoffset_x = 0;
   hist.qoffset_y = 0;
   hist.qoffset_z = 0;
   hist.srow_x = zeros(1,4);
   hist.srow_y = zeros(1,4);
   hist.srow_z = zeros(1,4);
   hist.intent_name = '';
   hist.magic = '';
   hist.originator = origin;
   
   return;					% data_history

   
   %% function to save a nifti
   
function save_nii(nii, fileprefix, old_RGB)
   
   if ~exist('nii','var') | isempty(nii) | ~isfield(nii,'hdr') | ...
	~isfield(nii,'img') | ~exist('fileprefix','var') | isempty(fileprefix)

      error('Usage: save_nii(nii, filename, [old_RGB])');
   end

   if isfield(nii,'untouch') & nii.untouch == 1
      error('Usage: please use ''save_untouch_nii.m'' for the untouched structure.');
   end

   if ~exist('old_RGB','var') | isempty(old_RGB)
      old_RGB = 0;
   end

   v = version;

   %  Check file extension. If .gz, unpack it into temp folder
   %
   if length(fileprefix) > 2 & strcmp(fileprefix(end-2:end), '.gz')

      if ~strcmp(fileprefix(end-6:end), '.img.gz') & ...
	 ~strcmp(fileprefix(end-6:end), '.hdr.gz') & ...
	 ~strcmp(fileprefix(end-6:end), '.nii.gz')

         error('Please check filename.');
      end

      if str2num(v(1:3)) < 7.1 | ~usejava('jvm')
         error('Please use MATLAB 7.1 (with java) and above, or run gunzip outside MATLAB.');
      else
         gzFile = 1;
         fileprefix = fileprefix(1:end-3);
      end
   end
   
   filetype = 1;

   %  Note: fileprefix is actually the filename you want to save
   %   
   if findstr('.nii',fileprefix) & strcmp(fileprefix(end-3:end), '.nii')
      filetype = 2;
      fileprefix(end-3:end)='';
   end
   
   if findstr('.hdr',fileprefix) & strcmp(fileprefix(end-3:end), '.hdr')
      fileprefix(end-3:end)='';
   end
   
   if findstr('.img',fileprefix) & strcmp(fileprefix(end-3:end), '.img')
      fileprefix(end-3:end)='';
   end

   write_nii(nii, filetype, fileprefix, old_RGB);

   %  gzip output file if requested
   %
   if exist('gzFile', 'var')
      if filetype == 1
         gzip([fileprefix, '.img']);
         delete([fileprefix, '.img']);
         gzip([fileprefix, '.hdr']);
         delete([fileprefix, '.hdr']);
      elseif filetype == 2
         gzip([fileprefix, '.nii']);
         delete([fileprefix, '.nii']);
      end;
   end;

   if filetype == 1

      %  So earlier versions of SPM can also open it with correct originator
      %
      M=[[diag(nii.hdr.dime.pixdim(2:4)) -[nii.hdr.hist.originator(1:3).*nii.hdr.dime.pixdim(2:4)]'];[0 0 0 1]];
      save([fileprefix '.mat'], 'M');
   end
   
   return					% save_nii


%-----------------------------------------------------------------------------------
function write_nii(nii, filetype, fileprefix, old_RGB)

   hdr = nii.hdr;

   if isfield(nii,'ext') & ~isempty(nii.ext)
      ext = nii.ext;
      [ext, esize_total] = verify_nii_ext(ext);
   else
      ext = [];
   end

   switch double(hdr.dime.datatype),
   case   1,
      hdr.dime.bitpix = int16(1 ); precision = 'ubit1';
   case   2,
      hdr.dime.bitpix = int16(8 ); precision = 'uint8';
   case   4,
      hdr.dime.bitpix = int16(16); precision = 'int16';
   case   8,
      hdr.dime.bitpix = int16(32); precision = 'int32';
   case  16,
      hdr.dime.bitpix = int16(32); precision = 'float32';
   case  32,
      hdr.dime.bitpix = int16(64); precision = 'float32';
   case  64,
      hdr.dime.bitpix = int16(64); precision = 'float64';
   case 128,
      hdr.dime.bitpix = int16(24); precision = 'uint8';
   case 256 
      hdr.dime.bitpix = int16(8 ); precision = 'int8';
   case 511,
      hdr.dime.bitpix = int16(96); precision = 'float32';
   case 512 
      hdr.dime.bitpix = int16(16); precision = 'uint16';
   case 768 
      hdr.dime.bitpix = int16(32); precision = 'uint32';
   case 1024
      hdr.dime.bitpix = int16(64); precision = 'int64';
   case 1280
      hdr.dime.bitpix = int16(64); precision = 'uint64';
   case 1792,
      hdr.dime.bitpix = int16(128); precision = 'float64';
   otherwise
      error('This datatype is not supported');
   end
   
   hdr.dime.glmax = round(double(max(nii.img(:))));
   hdr.dime.glmin = round(double(min(nii.img(:))));
   
   if filetype == 2
      fid = fopen(sprintf('%s.nii',fileprefix),'w');
      
      if fid < 0,
         msg = sprintf('Cannot open file %s.nii.',fileprefix);
         error(msg);
      end
      
      hdr.dime.vox_offset = 352;

      if ~isempty(ext)
         hdr.dime.vox_offset = hdr.dime.vox_offset + esize_total;
      end

      hdr.hist.magic = 'n+1';
      save_nii_hdr(hdr, fid);

      if ~isempty(ext)
         save_nii_ext(ext, fid);
      end
   else
      fid = fopen(sprintf('%s.hdr',fileprefix),'w');
      
      if fid < 0,
         msg = sprintf('Cannot open file %s.hdr.',fileprefix);
         error(msg);
      end
      
      hdr.dime.vox_offset = 0;
      hdr.hist.magic = 'ni1';
      save_nii_hdr(hdr, fid);

      if ~isempty(ext)
         save_nii_ext(ext, fid);
      end
      
      fclose(fid);
      fid = fopen(sprintf('%s.img',fileprefix),'w');
   end

   ScanDim = double(hdr.dime.dim(5));		% t
   SliceDim = double(hdr.dime.dim(4));		% z
   RowDim   = double(hdr.dime.dim(3));		% y
   PixelDim = double(hdr.dime.dim(2));		% x
   SliceSz  = double(hdr.dime.pixdim(4));
   RowSz    = double(hdr.dime.pixdim(3));
   PixelSz  = double(hdr.dime.pixdim(2));
   
   x = 1:PixelDim;

   if filetype == 2 & isempty(ext)
      skip_bytes = double(hdr.dime.vox_offset) - 348;
   else
      skip_bytes = 0;
   end

   if double(hdr.dime.datatype) == 128

      %  RGB planes are expected to be in the 4th dimension of nii.img
      %
      if(size(nii.img,4)~=3)
         error(['The NII structure does not appear to have 3 RGB color planes in the 4th dimension']);
      end

      if old_RGB
         nii.img = permute(nii.img, [1 2 4 3 5 6 7 8]);
      else
         nii.img = permute(nii.img, [4 1 2 3 5 6 7 8]);
      end
   end

   if double(hdr.dime.datatype) == 511

      %  RGB planes are expected to be in the 4th dimension of nii.img
      %
      if(size(nii.img,4)~=3)
         error(['The NII structure does not appear to have 3 RGB color planes in the 4th dimension']);
      end

      if old_RGB
         nii.img = permute(nii.img, [1 2 4 3 5 6 7 8]);
      else
         nii.img = permute(nii.img, [4 1 2 3 5 6 7 8]);
      end
   end

   %  For complex float32 or complex float64, voxel values
   %  include [real, imag]
   %
   if hdr.dime.datatype == 32 | hdr.dime.datatype == 1792
      real_img = real(nii.img(:))';
      nii.img = imag(nii.img(:))';
      nii.img = [real_img; nii.img];
   end

   if skip_bytes
      fwrite(fid, zeros(1,skip_bytes), 'uint8');
   end

   fwrite(fid, nii.img, precision);
%   fwrite(fid, nii.img, precision, skip_bytes);        % error using skip
   fclose(fid);

   return;					% write_nii


