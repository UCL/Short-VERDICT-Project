function sim_de_seq

% 62, +31 - -30
% SENSE 2   31 profiles "17 to -13"?
% half scan 62.5% -> 17 to -4
ny = 62 ;
nx = 68 ;
nz = 5 + 6 + 1 ;
ksp = zeros([ny nx nz]);

kzoff = 7 ;
kyoff = 9 ;

img  = zeros(size(ksp)) ;
img(10:50,20:50,2:11) = 1 ;

k_orig = i2k(img) ;

sd = epg_t1ffe('FA',15,'TR',0.01,'T1',1,'incr',150,'Ntr',12*22) ;

ic = 0 ;
for kz = [5:-1:-6]
   for ky = [-8:2:34]
        ic = ic + 1 ;
        ksp(ky+kyoff, :, kz+kzoff) = sd(ic) * k_orig(ky+kyoff, :, kz+kzoff) ; 
        ksp(ky+kyoff+1, :, kz+kzoff) = sd(ic) *k_orig(ky+kyoff+1, :, kz+kzoff) ; 
    end
end

eshow(k2i(ksp))
