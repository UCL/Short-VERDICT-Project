function pc_plot(dinfoRL, dinfoAP, dinfoFH)

ctrig = unique([dinfoRL.CardiacTriggerDelayTime]) ;

nphase = length(ctrig) ;

[vpcrl, mrl] = d2mat(dinfoRL,{'slice','ctdt','itype'},'itype',9,'op','dv') ;

[vpcap, map] = d2mat(dinfoAP,{'slice','ctdt','itype'},'itype',9,'op','dv') ;

[vpcfh, mfh] = d2mat(dinfoFH,{'slice','ctdt','itype'},'itype',9,'op','dv') ;

 tp = 18 ;

[ny nx nz nt] = size(vpcrl) ;

cvel = zeros([ny nx nz 3]) ;
fa = zeros([ny nx nz]) ;

maxv = 70

for ix = 1:nx
    for iy = 1:ny
        for iz = 1:nz 
            cvel(iy,ix,iz,1) = vpcrl(iy,ix,iz,tp)/maxv ;
            cvel(iy,ix,iz,2) = vpcap(iy,ix,iz,tp)/maxv ;
            cvel(iy,ix,iz,3) = vpcfh(iy,ix,iz,tp)/maxv ;
        
            fa(iy,ix,iz) = norm(squeeze(cvel(iy,ix,iz,:))) ;
        end
        
    end
end
        
rgb = vec2col(cvel,fa,0,1) ;
rgb(isnan(rgb))=0;
vdims = mrl.vdims(1:3) ;
eshow(sum(rgb,3),'vdim',vdims,'isrgb',true)

mip_rgb(rgb,fa)

eshow(rgb,'vdim',vdims,'isrgb',true)



