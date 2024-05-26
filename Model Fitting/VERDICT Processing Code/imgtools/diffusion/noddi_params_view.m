function noddi_params_view(paramsfile)
% noddi_params_view(paramsfile)
% Displays the output from NODDI processing
%
% Example usage:
%   noddi_params_view(paramsfile)
%
% Adapted from code by Gary Hui Zhang (gary.zhang@ucl.ac.uk)
%
% See also DICOM2NODDI NODDI_BATCH_FITTING
%

% load the fitting results
fprintf('loading the fitted parameters from : %s\n', paramsfile);
load(paramsfile);
% Loads 'model', 'gsps', 'fobj_gs', 'mlps', 'fobj_ml', 'error_code', 'sz_data', 'mask_locs'

total = sz_data(1) * sz_data(2) * sz_data(3) ;

% determine the volumes to be saved
% the number of fitted parameters minus the two fibre orientation
% parameters.
idxOfFitted = find(model.GD.fixed(1:end-2)==0);
noOfVols = length(idxOfFitted);
vols = zeros(total,noOfVols);
% the volume for the objective function values
fobj_ml_vol = zeros(sz_data);
% the volume for the error code
error_code_vol = zeros(sz_data);
% some special cases
if (strfind(model.name, 'Watson'))
    idxOfKappa = find(ismember(model.paramsStr, 'kappa')==1);
    odi_vol = zeros(sz_data);
end

% the volumes for the fibre orientations
fibredirs_x_vol = zeros(sz_data);
fibredirs_y_vol = zeros(sz_data);
fibredirs_z_vol = zeros(sz_data);
% compute the fibre orientations from the estimated theta and phi.
fibredirs = GetFibreOrientation(model.name, mlps);

% determine the volumes with MCMC fitting to be saved
if (model.noOfStages==3)
    idxOfFittedMCMC = find(model.MCMC.fixed(1:end-2)==0);
    noOfVolsMCMC = length(idxOfFittedMCMC);
    volsMCMC = zeros(total,noOfVolsMCMC);

    if (strfind(model.name, 'Watson'))
        idxOfKappa = find(ismember(model.paramsStr, 'kappa')==1);
	     if (model.MCMC.fixed(idxOfKappa)==0)
            odi_volMCMC = zeros(total,1);
        end
    end
end

% convert to volumetric maps
fprintf('converting the fitted parameters into volumetric maps ...\n');

% fitted parameters other than the fiber orientations
for j=1:length(idxOfFitted)
    vols(mask_locs,j) = mlps(:, idxOfFitted(j));
end
vols = reshape(vols,[sz_data(1) sz_data(2) sz_data(3) length(idxOfFitted)]) ;

% objective function values
fobj_ml_vol(mask_locs) = fobj_ml(:);

% error codes
error_code_vol(mask_locs) = error_code(:);

% fiber orientations
fibredirs_x_vol(mask_locs) = fibredirs(1,:)';
fibredirs_y_vol(mask_locs) = fibredirs(2,:)';
fibredirs_z_vol(mask_locs) = fibredirs(3,:)';
    
% special cases
if (strfind(model.name, 'Watson'))
    odi_vol(mask_locs) = atan2(1, mlps(:,idxOfKappa)*10)*2/pi;
end

% figure
% subplot(3,1,1), imshow(cat(2,fibredirs_x_vol, fibredirs_y_vol, fibredirs_z_vol))
% title('Fibre x,y,z')
% subplot(3,1,2), imshow(odi_vol)
% title('ODI')
% subplot(3,1,3), imshow(cat(2,vols(:,:,1), vols(:,:,2), vols(:,:,3)))
% title( [model.paramsStr{idxOfFitted(1)}, ', ', ...
%       model.paramsStr{idxOfFitted(2)}, ', ', ...
%       model.paramsStr{idxOfFitted(3)} ])

eshow(odi_vol,'name','ODI')
eshow(vols(:,:,:,1),'name',model.paramsStr{idxOfFitted(1)})
eshow(vols(:,:,:,2),'name',model.paramsStr{idxOfFitted(2)})
eshow(vols(:,:,:,3),'name',model.paramsStr{idxOfFitted(3)})

% Copied from noddi_qa
ny = sz_data(1) ; nx = sz_data(2) ; nz = sz_data(3) ;
ngrad = size(roi,2) ;

vol_in = zeros([ ny*nx*nz ngrad]) ;

vol_in(mask_locs,:) = roi ;
vol_in = reshape(vol_in,[ny nx nz ngrad]) ;


[D,S0] = dwi2tensor(vol_in,bvec,bval) ;


dradial = zeros(size(D,1),size(D,2),size(D,3)) ;
daxial = zeros(size(D,1),size(D,2),size(D,3)) ;
for id1 = 1:size(D,1)
    for id2 = 1:size(D,2)
        for id3 = 1:size(D,3)
            [evec, eval] = d2eig(D(id1,id2,id3,:,:)) ;
            if sum(eval) > realmin && isfinite(sum(eval))
                dradial(id1,id2,id3) = (eval(2)+eval(3)) /2;
                daxial(id1,id2,id3) = eval(1);
            else
                dradial(id1,id2,id3) = 0 ;
                daxial(id1,id2,id3) = 0 ;
            end
        end
    end
end
       
       
cfa = d2cfa(D) ;

eshow(cfa,'isrgb',1,'name','cfa') 
eshow(S0,'name','Fitted S0')
eshow(invariantsb(D,'fa'),'name','FA')
eshow(1e6*invariantsb(D,'mean'),'name','mean')
eshow(1e6*dradial,'name','Radial Diffusivity')
eshow(1e6*daxial,'name','Axial Diffusivity')

%figure('Name','Diffusivites Mean, radial, axial')
%imshow(1e6*[invariantsb(D,'mean') dradial daxial],[0 1500])




% for i=1:size(mlps,1)
%     % compute the index to 3D
%     volume_index = (idx(i,3)-1)*ysize*xsize + (idx(i,2)-1)*xsize + idx(i,1);
%   
% 	 % MCMC fitted parameters
%     if (model.noOfStages==3)
%         for j=1:length(idxOfFittedMCMC)
%             volsMCMC(volume_index,j) = mean(squeeze(mcmcps(i,:,j)));
%         end
%         if (strfind(model.name, 'Watson'))
%             % Warning: Here we hard-coded the index to kappa!!!
%             odi_volMCMC(volume_index) = atan2(1, mean(squeeze(mcmcps(i,:,3)))*10)*2/pi;
%         end
%     end
%     
% end

% save as NIfTI
% fprintf('Saving the volumetric maps of the fitted parameters ...\n');
% 
% niftiSpecs.dim = [xsize ysize zsize];
% niftiSpecs.mat = target.mat;
% niftiSpecs.mat_intent = target.mat_intent;
% niftiSpecs.mat0 = target.mat0;
% niftiSpecs.mat0_intent = target.mat0_intent;
% 
% % the fitted parameters other than the fiber orientations
% for i=1:length(idxOfFitted)
%     output = [outputpref '_' cell2mat(model.paramsStr(idxOfFitted(i))) '.nii'];
%     SaveAsNIfTI(reshape(squeeze(vols(:,i)), [xsize ysize zsize]), niftiSpecs, output);
% end
% 
% % the special cases
% if (strfind(model.name, 'Watson'))
%     output = [outputpref '_' 'odi.nii'];
%     SaveAsNIfTI(reshape(odi_vol, [xsize ysize zsize]), niftiSpecs, output);
% end
% 
% % the objective function values
% output = [outputpref '_' 'fmin.nii'];
% SaveAsNIfTI(reshape(fobj_ml_vol, [xsize ysize zsize]), niftiSpecs, output);
% 
% % the error codes
% output = [outputpref '_' 'error_code.nii'];
% SaveAsNIfTI(reshape(error_code_vol, [xsize ysize zsize]), niftiSpecs, output);
% 
% % the fibre orientations
% output = [outputpref '_' 'fibredirs_xvec.nii'];
% SaveAsNIfTI(reshape(fibredirs_x_vol, [xsize ysize zsize]), niftiSpecs, output);
% output = [outputpref '_' 'fibredirs_yvec.nii'];
% SaveAsNIfTI(reshape(fibredirs_y_vol, [xsize ysize zsize]), niftiSpecs, output);
% output = [outputpref '_' 'fibredirs_zvec.nii'];
% SaveAsNIfTI(reshape(fibredirs_z_vol, [xsize ysize zsize]), niftiSpecs, output);
% 
% % the MCMC fitted parameters
% if (model.noOfStages==3)
%     for i=1:length(idxOfFittedMCMC)
%         output = [outputpref '_' cell2mat(model.paramsStr(idxOfFittedMCMC(i))) '_MCMC.nii'];
%         SaveAsNIfTI(reshape(squeeze(volsMCMC(:,i)), [xsize ysize zsize]), niftiSpecs, output);
%     end
%     if (strfind(model.name, 'Watson'))
%         output = [outputpref '_' 'odi_MCMC.nii'];
%         SaveAsNIfTI(reshape(odi_volMCMC, [xsize ysize zsize]), niftiSpecs, output);
%     end
% end

