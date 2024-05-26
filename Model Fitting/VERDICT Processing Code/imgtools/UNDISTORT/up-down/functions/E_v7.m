function [u] = E_v7(v,arg)
% Encoding function with dB0 phase term and coil sensitivities
%
%inputs
S        = arg.S;
t_s      = arg.t_s;
dB0_rads = arg.dB0_rads;
kx       = arg.k.x;
ky       = arg.k.y;
offset  = arg.offset;
[sy,sx,nch,~] = size(S);

if isfield(arg,'samp')
    samp = arg.samp;
else
    samp = ones(sy,sx);
end

%initialise
u = zeros(sy,sx,nch);

%k-space co-ord system
x_coord = (1:sx) - (sx / 2) - 1; %orignal
y_coord = (1:sy) - (sy / 2) - 1; %original 
% x_coord = (1:sx) - floor(sx / 2) - 1;
% y_coord = (1:sy) - floor(sy / 2) - 1;
x_coord = -floor(sx/2):ceil(sx/2)-1;
y_coord = -floor(sy/2):ceil(sy/2)-1;

        tic
        num_time_points = numel(t_s);
        x_coord_mat = repmat(x_coord, [sy 1]);
        y_coord_mat = repmat(y_coord(:),[1 sx]);
        y_coord_mat = y_coord_mat(:);
        x_coord_mat = x_coord_mat(:);
        dB0_rads = dB0_rads(:);
        
        u = zeros(sy*sx,nch);
        size(v)
        size(S)
        
      %  v=v./(sqrt(sum(abs(S).^2,3)+0.1)); %ori
    %  v=v./((sum(abs(S).^2,3)+eps));
        v(isnan(v))=0;
%          v=v./(sqrt(sum(abs(S).^2,3))+eps);
%     
%          v(isnan(v))=0;
        
       % v = v .* S;  %usman
        for chn = 1:nch
            v_c=v.*S(:,:,chn);%usman
            %v_c = v(:,:,chn);%usman
            v_c = v_c(:);
            parfor ii = 1:num_time_points %par
                % Phase due to encoding gradients and B0 inhomogeneities
                phase_pix = ( kx(ii) .* x_coord_mat ) + ( ky(ii) .* (y_coord_mat+offset) ) + ( t_s(ii) .* dB0_rads ); %   par            
                
                sol =  v_c.' * exp( -1i .* phase_pix ); 
                
                
             %   sol=sol*sinc(kx(ii))*sinc(ky(ii));
                
                u(ii,chn) = sol;
            end
        end
        u = reshape(u,[sy sx nch]);
        toc
    
        

% Fix intensities (due to summation)
u = u ./ sqrt(sx .* sy);

%u = u ./ sqrt(repmat(sum((S.*conj(S))+eps,3),[1 1 nch]));

%multiply by sampling matrix
u = repmat(samp,[1 1 nch]) .* u;
end