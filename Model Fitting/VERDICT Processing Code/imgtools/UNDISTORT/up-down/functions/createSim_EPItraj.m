function [phi_x phi_y]=createSim_EPItraj(sz_FE,sz_PE,MR)
method=2; %3 for MR
switch method
    case 1
        %trajectory simulation
        f.traj = 'epi-under';
        f.dens = 'voronoi';
        samp=ones(sz_PE,1);
        N = [sz_FE sz_PE]; FOV=N;
        ig = image_geom('nx', sz_FE,'ny',sz_PE, 'fov', FOV); % 250 mm FOV
        [kspace omega wi] = mri_trajectory(f.traj, {'samp', samp}, N, ig.fov, {f.dens});
        
        x_mat = reshape(omega(:,2),sz_PE,sz_FE);
        y_mat = reshape(omega(:,1),sz_PE,sz_FE);
        
        y_mat2 = reshape(omega(:,1),sz_PE,sz_FE);
        y_mat2(:,2:2:end) = y_mat2(end:-1:1,2:2:end);
        
        omega(:,1)=y_mat(:);
        phi_x=omega(:,2);
        phi_y=omega(:,1);
        
    case 2
        % If data is already flipped, as is the case when using MRecon
        FE_idx = (([0:sz_FE-1]/sz_FE) - 0.5)*2*pi; %original
        PE_idx = (([0:sz_PE-1]/sz_PE)- 0.5)*2*pi;  % original
        
        
        FE_idx=((-ceil(sz_FE/2):floor(sz_FE/2)-1)/(sz_FE))*2*pi;
        PE_idx=((-ceil(sz_PE/2):floor(sz_PE/2)-1)/(sz_PE))*2*pi;
        
     
        
         
        [FE_mat PE_mat] = ndgrid(FE_idx, PE_idx);
        phi_x=PE_mat(:);
        phi_y=FE_mat(:);  
        
    case 3
        PE_idx = ([0:sz_PE-1]/sz_PE- 0.5)*2*pi;
        
        FE_vals = MR.Parameter.Labels.NusEncNrs;
                
        cntr_pnt = ceil(length(FE_vals)/2);
        idx = round((-sz_FE:2:(sz_FE-1))/2) + cntr_pnt;        
        FE_idx = FE_vals(idx);
        FE_idx = (FE_idx ./ (numel(FE_idx))) * 2 * pi;
        
        [FE_mat PE_mat] = ndgrid(FE_idx, PE_idx);
        phi_x=PE_mat(:);
        phi_y=FE_mat(:);  
        
end




end