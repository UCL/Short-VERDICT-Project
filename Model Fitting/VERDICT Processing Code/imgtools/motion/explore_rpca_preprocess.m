

%!!! Current issue is still scaling - suggest use the image median as a
%reference. Though median seems a bit low.

% D:\data\small bowel\Alex_RPCAdata\Anonymized - 0308012\SMALL BOWEL\BTFE 6x5mm 2 - 1401
% Frames 11:26 have little breathing no Busco?

% Only first few cells of this file are valid

dinfo = datparse ;
[Images, outp] = d2mat(dinfo,{'dyn','slice'},'slice',1,'resize',[128 128],'op','pv_single') ;

outp.vdims = [1 1 1 1] ;

[L S] = RPCA(Images, 2) ;

%high lambdaC to prevent it finding contrast changes
[LUx LUy LC LImagesReg LJacobianStd]=opt_flow(L,[1 1 1 1], 'lambdaU',1e-4 ,'lambdaC',1e-4) ;

cImages = apply_def(LUx, LUy, Images) ;

[cL cS] = RPCA(cImages) ;  % better S?


% Below is for looking at effects of scaling etc.
% [Ux Uy C ImagesReg JacobianStd]=opt_flow( Images,outp.vdims, 'pcm1',0,'pcm2',0,'lambdaU',0, 'lambdaC', 0 ) ;

%[nUx nUy nC nImagesReg nJacobianStd]=opt_flow( Images,outp.vdims, 'normalise',1 ) ;

% [ncUx ncUy ncC ncImagesReg ncJacobianStd]=opt_flow( single(Images)/2,outp.vdims, 'pcm1',0,'pcm2',0,...
%    'lambdaU', (1.0e-4)*2, 'lambdaC', 0 ) ;


%%
[L S] = RPCA(Images, 2) ;


[LUx LUy LC LImagesReg LJacobianStd]=opt_flow(L,outp.vdims, 'lambdaU',1e-4 ,'lambdaC',1e-6) ;

cImages = apply_def(LUx, LUy, Images) ;

%%

[cUx cUy cC cImagesReg cJacobianStd]=opt_flow( cImages,outp.vdims ) ;


%%

% DO NOT SCALE TO [0 1] 
%Images = Images / max(Images(:)) ;

[Ny Nx Nt] = size(Images) ;
dy = outp.vdims(1) ; 
dx = outp.vdims(2) ; 
dt = outp.vdims(3) ; 

tdim = 3 ;

% Find image closest to the median image
Imedian = median(Images,tdim);
dist    = zeros(1,Nt);
for t=1:Nt
    It = Images(:,:,t);
%     dist(t) = 1 - corr2(It, Imedian);
    dist(t) = norm( It(:) - Imedian(:) );
end
[val ref] = min(dist); % the reference image for registration
disp(['ref: ',num2str(ref)])


% Below from Freddy's Crohn's demo

% Do registration
fprintf('Starting registration...\n')

%-- Choose kernel size for image interpolations --
% W = 2; % linear interpolation
W = 6; % windowed sinc (lanczos window) interpolation

nbins   = 2^16;
LookupTable = zeros(nbins,1);
xList = -W/2 : W/(nbins-1) : W/2 ;
if(W==2)
    LookupTable = 1 - abs(xList);
else
    LookupTable = sinc(xList) .* sinc( xList / (W/2) ); % lanczos-sinc
end
% figure, plot( xList, LookupTable ), grid
Interpolator.LookupTable = LookupTable;
Interpolator.nbins       = nbins;
Interpolator.W           = W;
% figure, plot(Interpolator.LookupTable)


%-- Registration parameters --
% Param.lambdaU   = 1.e-3; % smoothness weight of the displacement fields
% Param.lambdaI   = 4.e-5; % smoothness weight of the map of intensity changes

Param.lambdaC   = 5e-6; % smoothness weight of the map of intensity changes
Param.lambdaI = Param.lambdaC ;
Param.lambdaU   = 1.e-4; % smoothness weight of the displacement fields
% Param.lambdaU   = 1.e-6; % smoothness weight of the displacement fields

Param.display   = 1;
Param.dx        = dx; % spatial resolution in mm/pixel
Param.dt        = dt; % temporal resolution in s/pixel
 Param.V0        = Inf; % mm/s
%  Param.V0        = 20; % mm/s
Param.ref       = ref;
Param.ResolutionLevels	= [1/8 1/4 1/2 1];

Param.NbLoops           = 4;
Param.Interpolator      = Interpolator;
Param.RegularizerOrder  = 2; % 1: minimize norm of the 1st order derivative of displacements and intensity maps
                             % 2: minimize norm of the 2nd order derivative of displacements and intensity maps


%-- Run registration --
% dummy run for memory allocation
% ParamD = Param ;
% ParamD.dummy = 1 ; ParamD.display = 0 ; ParamD.NbLoops = 1; ParamD.ResolutionLevels	= [1 1];
% gen_optical_flow_2D(Images,ParamD) ;

tic;
[Ux Uy C ImagesReg] = gen_optical_flow_2D(Images, Param);
elapsed_time = toc;

elapsed_time_min = floor(elapsed_time/60);
elapsed_time_sec = elapsed_time - elapsed_time_min*60;
fprintf('Total time: %d min %2.1f sec\n', ...
    elapsed_time_min, elapsed_time_sec );


%%

eshow(cat(2,double(Images),ImagesReg,C))

for t=1:Nt
    Ut(:,:,1,1) = Uy(:,:,t);
    Ut(:,:,1,2) = Ux(:,:,t);
    Jacobian(:,:,t) = jacobianDet(Ut);
end
Jacobian(1:2,:,:)       = 1;
Jacobian(end-1:end,:,:) = 1;
Jacobian(:,1:2,:)       = 1;
Jacobian(:,end-1:end,:) = 1;

JacobianStd         = std( Jacobian, 0, 3 );

eshow(JacobianStd)
    

%% Display registration results
M = mean(Images(:)) + 5*std(Images(:));

figure
for t=1:Nt
    subplot(231), imshow( Images(:,:,t), [0 M] )
    title( sprintf('Unregistered, t=%d', t) )
    
    subplot(232), imshow( ImagesReg(:,:,t), [0 M] )
    title( sprintf('Registered, t=%d', t) )
    
    subplot(233), imshow( ImagesReg(:,:,t)+C(:,:,t), [0 M] )
    title( sprintf('Registered and intensity corrected, t=%d', t) )
    
    subplot(235), imshow( ImagesReg(:,:,t)-Images(:,:,Param.ref), [-M M]/2 )
    title( sprintf('Registered, t=%d', t) )
    
    subplot(236), imshow( ImagesReg(:,:,t)+C(:,:,t)-Images(:,:,Param.ref), [-M M]/2 )
    title( sprintf('Reg + Int, t=%d', t) )
    
    pause(0.05)
end


return
%% previous code for 3D

%-- Choose kernel size for image interpolations --
% W = 2; % linear interpolation
W = 4; % windowed sinc (lanczos window) interpolation

nbins   = 2^16;
LookupTable = zeros(nbins,1);
xList = -W/2 : W/(nbins-1) : W/2 ;
if(W==2)
    LookupTable = 1 - abs(xList);
else
    LookupTable = sinc(xList) .* sinc( xList / (W/2) ); % lanczos-sinc
end
% figure, plot( xList, LookupTable ), grid
Interpolator.LookupTable = LookupTable;
Interpolator.nbins       = nbins;
Interpolator.W           = W;
% figure, plot(Interpolator.LookupTable)


tic;
for t=1:Nt
    close all
    
    fprintf('*********************************************\n');
    
    clear images
    images(:,:,:,1) = Images(:,:,:,t);
    images(:,:,:,2) = Images(:,:,:,ref);
    
    %-- Registration parameters --
    Param.lambdaU   = 1.e-6; % smoothness weight of the displacement fields
    Param.lambdaI   = 1.e-5; % smoothness weight of the map of intensity changes
%     Param.lambdaI   = 1.e-4; % smoothness weight of the map of intensity changes
    Param.display   = 1;
    Param.dx        = dx; % spatial resolution in mm/pixel
    Param.dy        = dy; % spatial resolution in mm/pixel
    Param.dz        = dz; % spatial resolution in mm/pixel
    Param.dt        = dt; % temporal resolution in s/pixel
%     Param.dx        = 1; % spatial resolution in mm/pixel
%     Param.dy        = 1; % spatial resolution in mm/pixel
%     Param.dz        = 1; % spatial resolution in mm/pixel
%     Param.dt        = 1; % temporal resolution in s/pixel
    % Param.V0        = Inf; % mm/s
    % Param.ref       = ref;
    Param.ref       = 2;
  %   Param.ResolutionLevels	= [1/8 1/4 1/2 1];
  Param.ResolutionLevels	= [ 1/2 1];
    Param.NbLoops           = 6;
    Param.Interpolator      = Interpolator;
    Param.RegularizerOrder  = 2; % 1: minimize norm of the 1st order derivative of displacements and intensity maps
                             % 2: minimize norm of the 2nd order derivative of displacements and intensity maps

    %-- Run registration --
    start_time = toc;
    [ux uy uz c imagesReg] = generalized_optical_flow_3D( images, Param);
    elapsed_time = toc - start_time;

    elapsed_time_min = floor(elapsed_time/60);
    elapsed_time_sec = elapsed_time - elapsed_time_min*60;
    fprintf('Registration complete for t=%d : time: %d min %2.1f sec\n', ...
        t, elapsed_time_min, elapsed_time_sec );

    ImagesReg(:,:,:,t)  = imagesReg(:,:,:,1);
    Ux(:,:,:,t)         = ux(:,:,:,1);
    Uy(:,:,:,t)         = uy(:,:,:,1);
    Uz(:,:,:,t)         = uz(:,:,:,1);
    C(:,:,:,t)          = c(:,:,:,1);
    
    if(0)
    
        z = 8;
        figure, imagesc( Images(:,:,z,t), [0 1] )
        figure, imagesc( ImagesReg(:,:,z,t), [0 1] )
        figure, imagesc( Images(:,:,z,ref), [0 1] )
        figure, imagesc( ImagesReg(:,:,z,t)+C(:,:,z,t), [0 1] )
        figure, imagesc( ImagesReg(:,:,z,t)+C(:,:,z,t) - Images(:,:,z,ref), [-1 1] )
        
        x = size(images,2)/2;
        figure, imagesc( squeeze(images(:,x,:,1) ), [0 1] )
        figure, imagesc( squeeze(imagesReg(:,x,:,1) ), [0 1] )
        figure, imagesc( squeeze(images(:,x,:,2) ), [0 1] )
        figure, imagesc( squeeze(imagesReg(:,x,:,1)+c(:,x,:,1) ), [0 1] )
        
        figure, imagesc(Uy(:,:,round(end/2),1))
        figure, imagesc(C(:,:,round(end/2),1))
        
    end
end
elapsed_time = toc;
 fprintf('Total time: %d min %2.1f sec\n', ...
        elapsed_time_min, elapsed_time_sec );

%% Display registration results

M = mean(Images(:)) + 5*std(Images(:));
figure(2)
z = 3;
for t=1:Nt
    subplot(231), imshow( Images(:,:,z,t), [0 M] )
    title( sprintf('Unregistered, t=%d', t) )
    
    subplot(232), imshow( ImagesReg(:,:,z,t), [0 M] )
    title( sprintf('Registered, t=%d', t) )
    
    subplot(233), imshow( ImagesReg(:,:,z,t)+C(:,:,z,t), [0 M] )
    title( sprintf('Registered and intensity corrected, t=%d', t) )
    
    subplot(234), imshow( Images(:,:,z,t)-Images(:,:,z,ref), [-M M]/2 )
    title( sprintf('Unregistered, t=%d', t) )
    
    subplot(235), imshow( ImagesReg(:,:,z,t)-Images(:,:,z,ref), [-M M]/2 )
    title( sprintf('Registered, t=%d', t) )
    
    subplot(236), imshow( ImagesReg(:,:,z,t)+C(:,:,z,t)-Images(:,:,z,ref), [-M M]/2 )
    title( sprintf('Registered & intensity corected, t=%d', t) )
    
    pause(0.02)
end

% save( 'workspace_bowel_motility_3D.mat' )











