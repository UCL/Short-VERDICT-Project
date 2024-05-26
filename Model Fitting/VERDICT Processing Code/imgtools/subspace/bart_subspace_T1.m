function bart_subspace_T1
% BART_SUBSPACE_T1 MATLAB version of webinar3 tutorial
%
% NOW LOOK INTO bart signal TSE
%

% Define Sequence Parameters
TR=0.006 ; % [s]
DIM=384 ; % Readouts
SPOKES=1 ; 
REP=700 ; % Repetitions
NC=8 ; % Number of Coils
BR=((DIM/2)) ; % Base resolution, as DIM includes two-fold oversampling

% create trajectory
temptraj = bart("traj -x "+DIM + " -y "+SPOKES + " -t "+REP + " -c -r -G") ;
traj     = bart("transpose 5 10", temptraj) ; 
traj1    = bart("scale 0.5", traj) ;

% check the dimensions of traj1
szinfo(traj1)

% create geometry basis functions
tempbasis_geom = bart("phantom -s"+NC+ " -T -k -b -t", traj1) ; 
szinfo(tempbasis_geom)

% create signal basis functions
tempbasis_simu_tubes = bart("signal -F -I -n"+REP + " -r"+TR + " -1 0.2:2.2:10 -2 0.045:0.045:1") ; 
tempbasis_simu_water = bart("signal -F -I -n"+REP + " -r"+TR + " -1 3:3:1      -2 1:1:1 ") ;

tempbasis_simu = bart("join 6",  tempbasis_simu_water, tempbasis_simu_tubes) ;
szinfo(tempbasis_simu)

% create simulated dataset
% bitmask 6  Here 6 is in a 0-based system
phantom_ksp = bart("fmac -s "+bitset(0,7), tempbasis_geom, tempbasis_simu) ;
szinfo(phantom_ksp)

% add noise to the simulated dataset 
phantom_ksp_noisy = bart("noise -n1", phantom_ksp) ; 

% generate a roi mask for later use
roi_mask = bart("phantom -x"+BR + " -T") ;
szinfo(roi_mask)

% sens = readcfl('/Users/davidatkinson/software/bart-webinars/webinar3/subspace_T1_intro/ref/sens_precomp') ;
% extract the steady-state data
nstate=300 ;
traj_state =         bart("extract 5 "+(REP-nstate) + " " + REP, traj) ; 
phantom_ksp_state =  bart("extract 5 "+(REP-nstate) + " " + REP, phantom_ksp_noisy) ;
traj_state1 =        bart("transpose 2 5 ", traj_state) ;
phantom_ksp_state1 = bart("transpose 2 5", phantom_ksp_state) ; 

% apply inverse nufft 
img = bart("nufft -i -d"+DIM + ":"+DIM + ":1 ", traj_state1, phantom_ksp_state1) ; 
    
% transform back to k-space and compute sensitivities
bsk012 = bitset(bitset(bitset(0,1),2),3) ;
ksp = bart("fft -u "+bsk012, img) ; 

% transpose because we already support off-center calibration region
% in dim 0 but here we might have it in 2
sens = bart("ecalib -S -t0.01 -m1", ksp) ; 
szinfo(sens)

% Generate the dictionary, perform SVD and
% Mss - (Mss + M0) * exp(-t * R1s)
nR1s=1000 ;
nMss=100  ;
dicc = bart("signal -F -I -n"+REP + " -r"+TR + " -1 5e-3:5:"+nR1s + " -3 1e-2:1:"+nMss) ; 
szinfo(dicc)

% # flags:
% #        -F FLASH
% #        -I Inversion recovery
% #        -n Number of RF excitations
% #        -r TR (in second)
% #        -1 Range for R1*: (5e-3, 5) [1/s]
% #        -3 Range for Mss: (1e-2, 1) * M0

% reshape the dicc to have all the elements on the 6th dimension
bmsk67 = bitset(bitset(0,7),8); % bitmask 6 7
dicc1 = bart("reshape "+bmsk67 + " " + (nR1s * nMss) + " 1", dicc) ;
szinfo(dicc1)

% squeeze the dicc1 before SVD
dicc2 = squeeze(dicc1) ;
szinfo(dicc2)

[U,S,V] = bart('svd -e', dicc2) ;
szinfo(U)

figure
tiledlayout('flow')
nexttile
plot(S(1:30)), ylabel('Singular values')

nexttile
plot(dicc2(:,[1 10 100 1000])), title('dicc2 samples')

nexttile
plot(U(:,1:5)), title('U first 5')

% create the temporal basis
nCoe=4 ; % use 4 coefficients
basis = bart("extract 1 0 "+nCoe , U) ; 
% transpose the basis to have time on the 6th dimension and coefficients on the 5th dimension
basis1 = bart("transpose 1 6 ", basis) ; 
basisnC = bart("transpose 0 5", basis1) ; 
szinfo(basisnC)

% 3.3.2 Perform subspace-constrained reconstruction with joint l1-Wavelet constraints on the coefficient maps
ITER=100 ;
REG=0.0005 ; 

% reconstruction with subspace constraint
bsk01 = bitset(bitset(0,1),2) ;
bsk6 = bitset(0,7) ;

cmd = "pics -e -d5 -i"+ITER + " -RW:"+bsk01 + ":" + bsk6 + ":"+REG + " -B basisnC -t traj" ;
writecfl('basisnC', basisnC)
writecfl('traj', traj)
subspace_reco = bart(cmd, phantom_ksp_noisy, sens) ; 

% resize the coefficient maps to the original size and multiply it with the roi_mask
tmp        = bart("resize -c 0 "+BR + " 1 "+BR, subspace_reco) ;
tmp_masked = bart("fmac ",roi_mask, tmp) ; 

% reshape the resutls for display
bsk16 = bitset(bitset(0,2),7) ;
subspace_maps = bart("reshape "+bsk16 + " " + (BR*nCoe) + " 1 ", tmp_masked) ; 
eshow(subspace_maps)

% multiply the basis with the subspace coefficient maps to obtain the projected images
imgs = bart("fmac -s "+bsk6 + " basisnC", tmp_masked) ;
% and show representative T1-weighted images
eshow(imgs)
