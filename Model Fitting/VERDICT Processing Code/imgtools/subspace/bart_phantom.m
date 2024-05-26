function bart_phantom
% From webinar 3
% https://github.com/mrirecon/bart-webinars/blob/master/webinar3/simu_phantom/tutorial_bart_simu_phantom.md
%

% Number of samples = Number of pixels
DIM=192 ;

comp_geom  = bart("phantom -x"+DIM + " -T -b") ; 
szinfo(comp_geom) % 192  192    1    1    1    1   11

% comp_geom is just 11, image-sized masks, one for each tube
% eshow(comp_geom)

szcg = size(comp_geom) ;

% bsk16 = bitset(bitset(0,7),2);
% comp_geom_flat = bart("reshape " + bsk16 + " " + ((DIM*GEOM_COMP)) + " 1 ", comp_geom) ;
% szinfo(comp_geom_flat)

comp_geom_flat = reshape(comp_geom,[szcg(1) szcg(2)*szcg(7)]) ;
szinfo(comp_geom_flat)

% water T1 3s, T2 1s
comp_water = bart("signal -F -I -r 0.0034 -n 400 -1 3:3:1 -2 1:1:1") ;
szinfo(comp_water)  % 1    1    1    1    1  400

% Tubes with three T1 values from 0.5s to 2s. T2 also waries but irrelevant
% here?
tcomp_tubes = bart("signal -F -I -r 0.0034  -n 400 -1 0.5:2:3 -2 0.005:0.2:3") ;
szinfo(tcomp_tubes) % 1    1    1    1    1  400    3    3

% bsk = bitset(bitset(0,7),8) ; 
% comp_tubes = bart("reshape " + bsk + " 9 1 ",tcomp_tubes) ;
% szinfo(comp_tubes)

comp_tubes = reshape(tcomp_tubes, [1 1 1 1 1 400 9]) ;

comp_simu = cat(7, comp_water, comp_tubes, comp_water) ;
szinfo(comp_simu)  % 1    1    1    1    1  400   11

figure('Name','signals')
plot(squeeze(comp_simu))

bsk6 = bitset(0,7) ;
phant = bart("fmac -s "+bsk6 + " ", comp_geom, comp_simu) ;
szinfo(phant)

% Here phant is image by reps ie 192 x 192 x .. x 400

% MATLABesque fmac
res = zeros([size(comp_geom,1) size(comp_geom,2) 1 1 1 400],'like',comp_geom) ;

for id7 = 1:11  % loops over tubes
    this_img = comp_geom(:,:,1,1,1,1,id7) ; % tube mask
    this_sig = comp_simu(1,1,1,1,1,:,id7) ; % signal for this tube
    
    res = res + repmat(this_img,[1     1 1 1 1 400 1]) .* ...
                repmat(this_sig,[192 192 1 1 1 1 1 1]) ;
end
szinfo(res) % 192  192    1    1    1  400

% K-SPACE version with non-Cartesian k-space phantom.
SAMPLES=96 ;
SPOKES=95 ;

% Create trajectory
ttraj = bart("traj -x "+(2*SAMPLES) + " -y "+SPOKES + " -r") ; 
bartinfo('traj',ttraj)

% Scale trajectory for 2-fold oversampling (halves k-space coord spacing)
traj = bart("scale 0.5", ttraj) ; 
bartinfo('traj',traj)

comp_geom_ksp = bart("phantom -T -b -k -t", traj) ;
szinfo(comp_geom_ksp)  %  1  192   95    1    1    1   11

% Transform the non-Cartesian k-space samples to image domain
comp_geom_img = bart("nufft -i -d "+DIM + ":" + DIM +":1", traj, comp_geom_ksp) ;
szinfo(comp_geom_img) % 192  192    1    1    1    1   11

% Back in k-space
phantom_ksp = bart("fmac -s "+bsk6, comp_geom_ksp, comp_simu) ;
szinfo(phantom_ksp) % 1  192   95    1    1  400





