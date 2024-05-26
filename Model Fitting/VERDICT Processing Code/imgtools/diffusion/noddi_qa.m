function noddi_qa(sl_out, bval, bvec, sz_slice, mask_locs)
% NODDI_QA Quality Assurance on NODDI (DTI)
%
%  noddi_qa(sl_out, bval, bvec, sz_slice, mask_locs)
%  noddi_qa
%
% To DO:
%  Read the volumes and process. Allow user selection of a pixel for
%  examination through frames for signal variation.
%  Play just B0's as a movie
%
% See also dicom2noddi
%

if nargin == 0
    disp(['Select all NODDI scans '])
    dinfo = datparse ;
    
    slices = 35
    
    if ~isfield(dinfo,'DiffusionDirectionality')
        error(['No DiffusionDirectionality - needs MultiFrame DICOM at the moment'])
    end
    
    
    [vol, matp, locs] = d2mat(dinfo,{'frame'},'slice', slices, 'op','fp') ;
    
    nf = size(vol,3) ;
    bval = zeros([nf 1]) ;
    bvec = zeros([nf 3]) ;
    
    
    hw = waitbar(0,['Processing ',num2str(nf),' frames']) ;
    wint = round(nf/100) ;
    
    loc_strip = [] ;
    
    for iframe = 1:nf
        if rem(nf,wint)== 0
            waitbar(iframe/nf,hw)
        end
        %tframe = vol(:,:,iframe) ;
        %sl_out(:,iframe) = tframe(mask_locs) ;
        
        bval(iframe) = dinfo(locs(iframe)).DiffusionBValue ;
        
        switch dinfo(locs(iframe)).DiffusionDirectionality
            case 0 % b=0
                bvec(iframe,:) = [0 0 0] ;
            case 1 % non-zero diffusion measurement
                bvec(iframe,:) = dinfo(locs(iframe)).DiffusionGradientOrientation ;
            case 2 % trace image etc
                % Need to strip this entry from the data
                loc_strip = [loc_strip iframe] ;
            otherwise
                error(['Un recognised DiffusionDirectionality'])
        end
    end
    close(hw)
    
    bval(loc_strip) = [] ;
    bvec(loc_strip,:) = [] ;
    vol(:,:,loc_strip) = [] ;
    
    jb0 = find(bval==0);
    jnon0 = find(bval > 0) ;
    
    volb0 = vol(:,:,jb0) ;
    B0 = sum(volb0,3) / size(volb0,3) ;
    
    volb  = vol(:,:,jnon0) ;
    bv = bval(jnon0) ;
    grad = bvec(jnon0,:) ;
    
    ny = size(volb,1) ; nx = size(volb,2) ; nz = length(slices) ;
    ngrad = length(bv) ;
    
    volb = reshape(volb,[ny nx nz ngrad]) ;
    D = dwi2tensor(volb, grad, bv, B0) ;
    
    
else
    
    
    ny = sz_slice(1) ; nx = sz_slice(2) ;
    if length(sz_slice)>2
        nz = sz_slice(3) ;
    else
        nz = 1;
    end
    
    ngrad = size(sl_out,2) ;
    
    sl_in = zeros([ ny*nx*nz ngrad]) ;
    
    sl_in(mask_locs,:) = sl_out ;
    sl_in = reshape(sl_in,[ny nx nz ngrad]) ;
    
    
    [D,S0] = dwi2tensor(sl_in,bvec,bval) ;
end

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
%eshow(S0,'name','S0')
eshow(1-invariantsb(D,'fa'),'name','1-FA')
eshow(1e6*invariantsb(D,'mean'),'name','mean')
eshow(1e6*dradial,'name','Radial Diffusivity')
eshow(1e6*daxial,'name','Axial Diffusivity')




