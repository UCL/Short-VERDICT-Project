%%%% Model based reconstruction 
%%%  with data  from phase encoding both UP and DOWN (PE AP/A and PE AP/P)
%%%  T2W as reference and B0 map from B0 scan were used for simulation
%%% same B0 used in the model for generation of measurements and for
%%% estimation





load D:\demo_basic_simulation_modelbased\ref_images %% T2W images taken as reference
load D:\demo_basic_simulation_modelbased\dB0_images %% dB0 in Hz, taken directly from scanner

islice=12;  %% perform B0 correction on slice number


I=ref_images(:,:,islice);
dB0=dB0_images(:,:,islice);
dB0=smoothn(dB0,2,'Robust'); %% smoothing applied to raw B0 to avoid noise spikes

[row col]=size(I);
opt.sx=row;
opt.sy=col;
opt.nch=1;%1
Sens=ones(row,col); %sensitivities




%% timing matrix
factor=2; %% scale of distortion, increase it to create more distortion
echo_spacing=factor*4e-4; %time between two samples along PE direction 
temp=[0:echo_spacing:(row-1)*echo_spacing]'; %128 samples in one read out
t_s_m1=repmat(temp,[1 col]);


dB0=dB0*2*pi;
no_dB0 = zeros(opt.sy,opt.sx);



% Simulate Cartesian Trajectory
display('creating traj')
[tr.x tr.y]=createSim_EPItraj(opt.sx,opt.sy);


%% distortion corrpted data in direction UP
arg1=struct('S',Sens,'t_s',t_s_m1,'dB0_rads',dB0,'k',tr);
Sk1=E_v4(I,arg1);


%% distortion corrupted data in direction DOWN
arg2=struct('S',Sens,'t_s',flipud(t_s_m1),'dB0_rads',dB0,'k',tr);
Sk2=E_v4(I,arg2);

Sk=[Sk1;Sk2];%concatenated data from two directions

%% preliminary model based reconstruction by using exact B0 map
tic;I_dis = Eh_v5(Sk,arg1,arg2);toc %



opt.num_it=12;
opt.r_init=zeros(opt.sy,opt.sx);
opt.recon_choice='cg';   % conjugate gradient
opt.Eh_fh = @Eh_v5;
opt.E_fh  = @E_v5;
%% CG reconstruction
I_undis1=Recon_withdB0_USM2(Sk,opt,arg1,arg2);
rec_final=I_undis1(:,:,end);%% final distortion corrected image




