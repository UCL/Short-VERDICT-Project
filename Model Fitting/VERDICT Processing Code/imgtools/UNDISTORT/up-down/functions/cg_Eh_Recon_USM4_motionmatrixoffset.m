function [ x, x_com ]= cg_Eh_Recon_USM4_motionmatrixoffset( yi, arg1,arg2, opt)
% conjugate gradient iterative recon
%
% Solve ::  (Eh*E) x = Eh y    ==   (A'A) x = A' b 
%
% D : preconditioning matrix    (reduce condition number for faster convergence)
% r : residual  (how far we are from the correct value of b)
% alpha :  size of step to take in direction of
% beta : search direction

precond   = 0; %seems to make worse when on
[sy sx nch ~]=size(yi);
sy=sy/2;

%handles to encoding functions
Eh = opt.Eh_fh;
E  = opt.E_fh;







dimensions=[sy sx nch];
if ~isfield(arg1,'num_it'), arg1.num_it = 20; end
if ~isfield(arg1,'r_init'), arg1.r_init = eps*randn(dimensions); end

if ~isfield(arg2,'num_it'), arg2.num_it = 20; end
if ~isfield(arg2,'r_init'), arg2.r_init = eps*randn(dimensions); end

x_store=zeros(sy,sx,arg1.num_it);

%preconditioning matrix
if precond
%     Sens=arg1.S;
%     D = sum(Sens .* conj(Sens),3);
%     D(D <= 0) = min(D(D>0));
%     D = D.*1./(sqrt(D)+0.05);
%     D(:,size(D,2)/4+1:size(D,2)*3/4)=0.0001;
%     D(:,1:size(D,2)/4)=1;
%     D(:,size(D,2)*3/4:end)=1;
   % D = D.*1./(sqrt(D)+0.05);%usman
   D=arg1.D;
else
    D=1; 
end



% if precond
%   D = arg1.dB0_rads;
%     D(D == 0) = min(D(D>0));
%     D = D.*1./(sqrt(D)+0.05);
% else
%     D=1; 
% end

% initialize fftshift
S = ones(sy, sx,nch);

%fftshift
yi=yi;

%----------------------------------------
%           BEGIN
%----------------------------------------
%initial guess at x
x = opt.r_init(:,:,1);

% Image from passing first guess (random noise) through encoding pipeline
x_res = Eh( E( x, arg1,arg2) , arg1,arg2);

% Initial residual: (rhs)
%  -- Difference between Eh of true data and EhE of first guess. 
%     If first guess is perfect r = 0;
r = Eh(yi,arg1,arg2) - x_res;
%r = sum(r,3);    %combine coils 

% Direction is now equal to this residual
ddir  = r.*D;
dnew  = scalar(r,r);
dorig = dnew;

%------------------
% iterate
%------------------
vec=[];
for it = 1:opt.num_it    
    display(sprintf('iteration : %d',it))        
    
    q = Eh( E( ddir, arg1,arg2), arg1,arg2);
    q = q .* D;    
    
    alpha = dnew / real(scalar(ddir,q));        %real?
    x = x + (alpha * ddir); %solution           
    r = r - (alpha * q);
    
    %search direction
    dold = dnew;
    dnew = (scalar(r,r));  
    ddir = r + ( (dnew ./ dold) * ddir );
        
    x_store(:,:,it) = x; %store iterations
    %if ( (dnew/dorig) < 1e-4),break,end    
    vec=[vec;(sqrt(dnew)/sqrt(dorig))];
end

x=x_store;

% postprocessing...
%Sens_corr = sum( conj(Sens) .* Sens ,3);
Sens_corr = ones(sy, sx);

for it=1:opt.num_it
    if precond
        x(:,:,it)= D.* x(:,:,it);
    end
    x_com(:,:,it) = x_store(:,:,it) .* (1./Sens_corr) .* S(:,:,1);
end

end

function v = scalar(a, b)
v = a(:)'*b(:);
end
